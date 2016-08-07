/*!
 * \brief multi-category classification
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>

#include <iostream>
#include <string>
#include <vector>
#include <fstream>

using namespace std;

#define MAX_STRING 100
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000
#define MAX_CODE_LENGTH 40

const int vocab_hash_size = 30000000;  // Maximum 30 * 0.7 = 21M words in the vocabulary.
const int category_hash_size = 30000000; // Maximum label size.

typedef float real;

struct document_category {
    long long cn;  // Frequency of label.
    int *point;  // Huffman tree (n leaf + n inner node, exclude root) path, (root, leaf], node index.
    char *name;  // Label name.
    char *code; // Huffman code, (root, leaf], 0/1 codes.
    char codelen;  // Huffman code length.
};

struct vocab_word {
    long long cn; // Frequency of word.
    char *word;  // word string.
};

char train_file[MAX_STRING], output_file[MAX_STRING];
char save_vocab_file[MAX_STRING], read_vocab_file[MAX_STRING];
int binary = 0, debug_mode = 2, min_count = 1, num_threads = 1, min_reduce = 1;
int *vocab_hash;  // Hash table.
struct vocab_word *vocab;  // Word vocabulary
long long vocab_max_size = 1000, vocab_size = 0, layer1_size = 10;
real alpha = 0.1, starting_alpha, sample = 0;
// syn0 => input document, syn1 => vitural label in the huffman tree.
real *syn0, *syn1, *expTable;
clock_t start;


struct Sentence {
    vector<string> words_;
    vector<string> ngrams_;
    string label_;
    vector<int32_t> features_;
    int size() { return features_.size(); }
};
vector<Sentence *> docs;

struct document_category *category;  // Label vocabulary.
int *category_hash; // Label hash table.
long long category_size = 0, category_max_size = 1000;

char ctrl_a = 1;  // Character ctrl-a
int epoch = 5;  // iteration number
real MIN_LR = 0.000001;
int ngram = 2;

//----------------------------------------------------
void InitUnigramTable() {
}

void Split(const string &s, const char *delims, vector<string> &res) {
    int ind_sep = s.find(delims);
    int start = 0;
    while (ind_sep != -1) {
        string t = s.substr(start, ind_sep-start);
        if (t != "") res.push_back(t);
        start = ind_sep + 1;
        ind_sep = s.find(delims, start);
    }
    res.push_back(s.substr(start, s.size() - start));
}

void ReadLabelInstance(const string &text, Sentence *sen) {
    int idx = text.find('\001');
    string label = text.substr(0, idx);
    string words = text.substr(idx+1);
    Split(words, " ", sen->words_);
    sen->label_ = label;
}

// Returns hash value of a word
int GetWordHash(const string &word) {
    unsigned long long a, hash = 0;
    for (a = 0; a < word.size(); a++) {
        hash = hash * 257 + word[a];
    }
    hash %= vocab_hash_size;
    return hash;
}

int AddNewLabel(const char *label) {
    unsigned int hash, length = strlen(label) + 1;
    if (length > MAX_STRING) length = MAX_STRING;
    category[category_size].name = (char *)calloc(length, sizeof(char));
    strcpy(category[category_size].name, label);
    category[category_size].cn = 0;
    category_size++;
    // Reallocate memory if needed
    if (category_size + 2 >= category_max_size) {
        category_max_size += 1000;
        category = (struct document_category *) realloc(category, category_max_size * sizeof(struct document_category));
    }
    hash = GetWordHash(label);
    while (category_hash[hash] != -1) hash = (hash + 1) % category_hash_size;
    category_hash[hash] = category_size - 1;
    return category_size - 1;
}

int AddWordToVocab(const char *word) {
    unsigned int hash, length = strlen(word) + 1;
    if (length > MAX_STRING) length = MAX_STRING;
    vocab[vocab_size].word = (char *) calloc(length, sizeof(char));
    strcpy(vocab[vocab_size].word, word);
    vocab_size++;
    // Reallocate memory if needed.
    if (vocab_size + 2 >= vocab_max_size) {
        vocab_max_size += 1000;
        vocab = (struct vocab_word*) realloc(vocab, vocab_max_size * sizeof(struct vocab_word));
    }
    hash = GetWordHash(word);
    while(vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
    vocab_hash[hash] = vocab_size - 1;
    return vocab_size - 1;
}

int SearchLabel(const char *label) {
    unsigned int hash = GetWordHash(label);
    while (1) {
        if (category_hash[hash] == -1) return -1;
        if (!strcmp(label, category[category_hash[hash]].name)) {
            return category_hash[hash];
        }
        hash = (hash + 1) % category_hash_size;
    }
}

// Returns position of a word in the vocabulary; if the word is
// not found, returns -1.
int SearchVocab(const char *word) {
    unsigned int hash = GetWordHash(word);
    while (1) {
        if (vocab_hash[hash] == -1) return -1;
        if (!strcmp(word, vocab[vocab_hash[hash]].word)) return vocab_hash[hash];
        hash = (hash + 1) % vocab_hash_size;
    }
}

// comparator document_category, for sorting.
int CategoryCompare(const void *a, const void *b) {
    return (int) (((struct document_category *) b)->cn - ((struct document_category *) a)->cn);
}

// Free memory space of vocab structure.
void DestroyCategory() {
    for (int a = 0; a < category_size; a++) {
        if (category[a].name != NULL) free(category[a].name);
        if (category[a].code != NULL) free(category[a].code);
        if (category[a].point != NULL) free(category[a].point);
    }
    free(category[category_size].name);
    free(category);
}

// Sorts the vocabulary by frequency using word counts.
void SortLabel() {
    int a, size;
    unsigned int hash;
    qsort(&category[0], category_size, sizeof(struct document_category),
          CategoryCompare);
    // Reset and compute hash.
    for (a = 0; a < category_hash_size; a++) category_hash[a] = -1;
    size = category_size;
    for (a = 0; a < size; a++) {
        if (category[a].cn < min_count) {
            category_size--;
            free(category[a].name);
            category[a].name = NULL;
        } else {
            // Hash wiil be re-computed, as after the sorting it is not actual.
            hash = GetWordHash(category[a].name);
            while (category_hash[hash] != -1) {
                hash = (hash + 1) % category_hash_size;
            }
            category_hash[hash] = a;
        }
    }

    category = (struct document_category *) realloc(category, (category_size + 1) *
                                                 sizeof(struct document_category));
    // Allocate memory for the binary tree construction.
    for (a = 0; a < category_size; a++) {
        category[a].code = (char *) calloc(MAX_CODE_LENGTH, sizeof(char));
        category[a].point = (int *) calloc(MAX_CODE_LENGTH, sizeof(int));
    }
}

// Reduces the vocabulary by removing infrequent category.
void ReduceCategory() {
    int a, b = 0;
    unsigned int hash;
    for (a = 0; a < category_size; a++) {
        if (category[a].cn > min_reduce) {
            category[b].cn = category[a].cn;
            category[b].name = category[a].name;
            b++;
        } else {
            free(category[a].name);
        }
    }
    category_size = b;
    for (a = 0; a < category_hash_size; a++) {
        category_hash[a] = -1;
    }
    for (a = 0; a < category_size; a++) {
        hash = GetWordHash(category[a].name);
        while (category_hash[hash] != -1) {
            hash = (hash + 1) % category_hash_size;
        }
        category_hash[hash] = a;
    }
    fflush(stdout);
    min_reduce++;
}

// Create binary Huffman tree using the category counts
// Frequent category will have short unique binary codes.
void CreateBinaryTree() {
  long long a, b, i, mini[2], pos1, pos2, point[MAX_CODE_LENGTH];
  char code[MAX_CODE_LENGTH];
  long long *count = (long long *) calloc(category_size * 2 + 1, sizeof(long long));
  long long *binary = (long long *) calloc(category_size * 2 + 1, sizeof(long long));
  long long *parent_node = (long long *) calloc(category_size * 2 + 1, sizeof(long long));
  for (a = 0; a < category_size; a++) count[a] = category[a].cn;
  for (a = category_size; a < category_size * 2; a++) count[a] = 1e15;
  pos1 = category_size - 1;
  pos2 = category_size;
  // Constructs the huffman tree by adding one node at a time.
  for (a = 0; a < category_size - 1; a++) {
    // Find two smallest nodes 'min1, min2'
    for (b = 0; b < 2; b++) {
        if (pos1 >= 0 && (count[pos1] < count[pos2]) ) {
            mini[b] = pos1;
            pos1--;
        } else {
            mini[b] = pos2;
            pos2++;
        }
    }
    count[category_size + a] = count[mini[0]] + count[mini[0]];
    parent_node[mini[0]] = category_size + a;
    parent_node[mini[1]] = category_size + a;
    binary[mini[1]] = 1; // right child set to 1.
  }
  // Now assign binary code to each vocabulary word.
  for (a = 0; a < category_size; a++) {
    b = a;
    i = 0;
    while (1) { // [leaf, root)
      code[i] = binary[b];
      point[i] = b;
      i++;
      b = parent_node[b];
      if (b == category_size * 2 - 2) break;
    }
    category[a].codelen = i;
    category[a].point[0] = category_size - 2;
    // Reverse, make it to be (root, leaf]
    for (b = 0; b < i; b++) {
      category[a].code[i - b - 1] = code[b];
      category[a].point[i - b] = point[b] - category_size;
    }
  }
  free(count);
  free(binary);
  free(parent_node);
}

void AddNGrams(Sentence *sen, int32_t ngram) {
    for (size_t i = 0; i < sen->words_.size() - ngram + 1; ++i) {
        string gram = "<-";
        for (int j = 0; j < ngram; ++j) {
            gram += sen->words_[i+j] + "-";
        }
        gram += ">";
        sen->ngrams_.push_back(gram);
    }
}

void LearnVocabFromTrainFile() {
    long long a, label_idx;
    // Initialize vocab hash table.
    for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
    for (a = 0; a < category_hash_size; a++) category_hash[a] = -1;
    ifstream fin(train_file);
    if (!fin.is_open()) {
        fprintf(stderr, "Error: training data file not found!\n");
        exit(1);
    }
    vocab_size = 0;
    string line;
    category_size = 0;
    while (getline(fin, line)) {
        Sentence *sen = new Sentence();
        ReadLabelInstance(line, sen);
        AddNGrams(sen, ngram);
        label_idx = SearchLabel(sen->label_.c_str());
        if (label_idx == -1) {  // New label
            a = AddNewLabel(sen->label_.c_str());
            category[a].cn = 1;
        } else {
            category[label_idx].cn++;
        }

        for (size_t i = 0; i < sen->words_.size(); ++i) {
            int wid = SearchVocab(sen->words_[i].c_str());
            if (wid == -1) { // New word
                wid = AddWordToVocab(sen->words_[i].c_str());
            }
            sen->features_.push_back(wid);
        }
        for (size_t i = 0; i < sen->ngrams_.size(); ++i) {
            int wid = SearchVocab(sen->ngrams_[i].c_str());
            if (wid == -1) {
                wid = AddWordToVocab(sen->ngrams_[i].c_str());
            }
            sen->features_.push_back(wid);
        }

        docs.push_back(sen);
    }

    SortLabel();

    if (debug_mode > 0) {
        fprintf(stderr, "Category size: %lld\n", category_size);
        fprintf(stderr, "Read [%d] Sentences\n", docs.size());
        fprintf(stderr, "Vocab contains [%lld] tokens\n", vocab_size);
    }

    fin.close();
}

void SaveVocab() {
}

void ReadVocab() {
}

void InitNet() {
    long long a, b;
    a = posix_memalign((void **) &syn0, 128,
                       vocab_size * layer1_size * sizeof(real));
    if (syn0 == NULL) {
        fprintf(stderr, "Syn0: memory allocation failed.\n");
        exit(1);
    }

    a = posix_memalign((void **) &syn1, 128,
                       category_size * layer1_size * sizeof(real));
    if (syn1 == NULL) {
        fprintf(stderr, "Syn1: memory allocation failed.\n");
        exit(1);
    }

    // Initialize weight parameters.
    for (b = 0; b < layer1_size; b++) {
        for (a = 0; a < category_size; a++) {
            syn1[a * layer1_size + b] = 0;
        }
    }

    // Random initialize each word's embedding.
    for (b = 0; b < layer1_size; b++) {
        for (a = 0; a < vocab_size; a++) {
            syn0[a * layer1_size + b] =
                    (real) ((rand() / (real)RAND_MAX - 0.5) / layer1_size);
        }
    }

    CreateBinaryTree();  // Create document label huffman tree.
}

void DestroyNet() {
    if (syn0 != NULL) free(syn0);
    if (syn1 != NULL) free(syn1);
}

void DestroyVocab() {
    for (int a = 0; a < vocab_size; a++) free(vocab[a].word);
    free(vocab);
}

void *TrainModelThread(void *id) {
    long long c, d, l2, last_word, last_label;
    long long word_count = 0;
    real f, g; // gradient

    // neu1 is hidden units, neu1e is hidden resual error.
    real *neu1 = (real *) calloc(layer1_size, sizeof(real));
    real *neu1e = (real *) calloc(layer1_size, sizeof(real));
    
    long long ntokens = 0;
    for (size_t i = 0; i < docs.size(); ++i) ntokens += docs[i]->size();

    for (int iter = 0; iter < epoch; ++iter) {
        for (size_t i = 0; i < docs.size(); ++i) {
            Sentence *sen = docs[i];
            word_count += sen->size();

            if (word_count % 10000 == 0) { // decay learning rate.
                real progress = (real) word_count / (ntokens * epoch);
                alpha = starting_alpha * (1.0 - progress);
                if (alpha < MIN_LR) alpha = MIN_LR;
                fprintf(stderr, "process: %.2f%%, lr:%.6f\r", (progress * 100), alpha);
            }
            for (c = 0; c < layer1_size; c++) neu1[c] = 0;
            for (c = 0; c < layer1_size; c++) neu1e[c] = 0;

            int total = 0;
            for (int j = 0; j < sen->size(); ++j) {
                // last_word = SearchVocab(sen->words_[j].c_str());
                last_word = sen->features_[j];
                if (last_word != -1) {
                    total++;
                    for (c = 0; c < layer1_size; c++) {
                        neu1[c] += syn0[c + last_word * layer1_size];
                    }
                }
            }
            for (c = 0; c < layer1_size; c++) neu1[c] /= total;  // average
        
            last_label = SearchLabel(sen->label_.c_str());

            if (last_label == -1) {
                cerr << "low frequent label:" << sen->label_ <<endl;
                continue;
            }

            // hs optimization
            for (d = 0; d < category[last_label].codelen; d++) {
                f = 0;
                l2 = category[last_label].point[d] * layer1_size;  // parent nodes in the Huffman tree.
                // Propagate hidden -> output
                for (c = 0; c < layer1_size; c++) f += neu1[c] * syn1[c + l2];

                if (f <= -MAX_EXP ) {
                    f = 0.0;
                } else if (f >= MAX_EXP) {
                    f = 1.0;
                } else {
                  f = expTable[(int) ((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
                }

                // 'g' is the gradient multiplied by the learning rate.
                g = (category[last_label].code[d] - f) * alpha;
                // Propagate errors output -> hidden.
                for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1[c + l2];
                // Learn weights hidden->output
                for (c = 0; c < layer1_size; c++) syn1[c + l2] += g * neu1[c];
            }

            for (c = 0; c < layer1_size; c++) neu1e[c] /= total;

            // Hidden -> input
            for (int j = 0; j < sen->size(); ++j) {
                // last_word = SearchVocab(sen->words_[j].c_str());
                last_word = sen->features_[j];
                if (last_word == -1) continue;
                for (c = 0; c < layer1_size; c++) {
                    syn0[c + last_word * layer1_size] += neu1e[c];
                }
            }
        }
    }
    free(neu1);
    free(neu1e);
    pthread_exit(NULL);
}

void TrainModel() {
    long a, b;
    FILE *fo;
    pthread_t *pt = (pthread_t *) malloc(num_threads * sizeof(pthread_t));
    if (pt == NULL) {
        fprintf(stderr, "Cannot allocate memory for threads\n");
        exit(1);
    }
    fprintf(stdout, "Starting training using file %s\n", train_file);
    starting_alpha = alpha;
    if (read_vocab_file[0] != 0) {
        ReadVocab();
    } else {
        LearnVocabFromTrainFile();
    }
    if (save_vocab_file[0] != 0) {
        SaveVocab();
    }
    if (output_file[0] == 0) return;

    InitNet();

    start = clock();
    for (a = 0; a < num_threads; a++) {
        pthread_create(&pt[a], NULL, TrainModelThread, (void *) a);
    }
    for (a = 0; a < num_threads; a++) {
        pthread_join(pt[a], NULL);
    }
    fflush(stdout);
    fprintf(stdout, "Training time %lds\n", (long)((clock() - start + 1)/(real)CLOCKS_PER_SEC));

    fo = fopen(output_file, "wb");
    if (fo == NULL) {
        fprintf(stderr, "Open %s failed.\n", output_file);
        exit(1);
    }

    // Save word vectors.
    fprintf(fo, "%lld %lld %lld\n", vocab_size, category_size, layer1_size);
    for (a = 0; a < vocab_size; a++) {
        if (vocab[a].word != NULL) {
            fprintf(fo, "%s ", vocab[a].word);
        }
        if (binary) {
            for (b = 0; b < layer1_size; b++) {
                fwrite(&syn0[a * layer1_size + b], sizeof(real), 1, fo);
            }
        } else {
            for (b = 0; b < layer1_size; b++) {
                fprintf(fo, "%lf ", syn0[a * layer1_size + b]);
            }
            fprintf(fo, "\n");
        }
    }

    // Save label weights and huffman codes;
    for (a = 0; a < category_size; a++) {
        if (binary) {
            for (b = 0; b < layer1_size; b++) {
                fwrite(&syn1[a * layer1_size + b], sizeof(real), 1, fo);
            }
        } else {
            for (b = 0; b < layer1_size; b++) {
                fprintf(fo, "%lf ", syn1[a * layer1_size + b]);
            }
            fprintf(fo, "\n");
        }
    }

    // Save category code, name, and point
    for (a = 0; a < category_size; a++) {
        fprintf(fo, "%s ", category[a].name);
        for (b = 0; b < category[a].codelen; b++) {
            fprintf(fo, "%d", category[a].code[b]);
        }
        fprintf(fo, " %d", category[a].point[0]);
        for (b = 1; b < category[a].codelen; b++) {
            fprintf(fo, ",%d", category[a].point[b]);
        }
        fprintf(fo, "\n");
    }

    fclose(fo);
    free(pt);
    DestroyCategory();
    DestroyVocab();
}

int ArgPos(const char *str, int argc, char **argv) {
    int a;
    for (a = 1; a < argc; a++) {
        if (!strcmp(str, argv[a])) {
            if (a == argc - 1) {
                fprintf(stderr, "Argument missing for %s\n", str);
                exit(1);
            }
            return a;
        }
    }
    return -1;  // Not found.
}

void Usage() {
    printf("multicategory classification toolkit v 0.1b\n\n");
    printf("Options:\n");
    printf("Parameters for training:\n");
    printf("\t-train <file>\n");
    printf("\t\tUse text data from <file> to train the model\n");
    printf("\t-output <file>\n");
    printf("\t\tUse <file> to save the resulting word vectors / word clusters\n");
    printf("\t-size <int>\n");
    printf("\t\tSet size of word vectors; default is 100\n");
    printf("\t-sample <float>\n");
    printf("\t\tSet threshold for occurrence of words. Those that appear with higher frequency");
    printf(" in the training data will be randomly down-sampled; default is 0 (off), useful value is 1e-5\n");
    printf("\t-hs <int>\n");
    printf("\t-threads <int>\n");
    printf("\t\tUse <int> threads (default 1)\n");
    printf("\t-min-count <int>\n");
    printf("\t\tThis will discard words that appear less than <int> times; default is 5\n");
    printf("\t-alpha <float>\n");
    printf("\t\tSet the starting learning rate; default is 0.025\n");
    printf("\nExamples:\n");
    printf("./doc2label -train data.txt -output model -size 10 \n\n");
}

int main(int argc, char **argv) {
    int i;
    if (argc == 1) {
        Usage();
        return 0;
    }

    output_file[0] = 0;
    save_vocab_file[0] = 0;
    read_vocab_file[0] = 0;
    if ((i = ArgPos("-size", argc, argv)) > 0) layer1_size = atoi(argv[i + 1]);
    if ((i = ArgPos("-train", argc, argv)) > 0) strcpy(train_file, argv[i + 1]);
    if ((i = ArgPos("-output", argc, argv)) > 0) strcpy(output_file, argv[i + 1]);
    if ((i = ArgPos("-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);

    starting_alpha = alpha;

    vocab = (struct vocab_word *) calloc(vocab_max_size, sizeof(struct vocab_word));
    vocab_hash = (int *) calloc(vocab_hash_size, sizeof(int));
    category = (struct document_category *) calloc(category_max_size, sizeof(struct document_category));
    category_hash = (int *) calloc(category_hash_size, sizeof(int));
    expTable = (real *) malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
    if (expTable == NULL) {
        fprintf(stderr, "out of memory\n");
        exit(1);
    }
    for (i = 0; i < EXP_TABLE_SIZE; i++) {
        real x = real(i * 2 * MAX_EXP) / EXP_TABLE_SIZE - MAX_EXP;
        expTable[i] = 1.0 / (1.0 + exp(-x));
    }
    TrainModel();
    DestroyNet();
    free(vocab_hash);
    free(category_hash);
    free(expTable);
    return 0;
}
