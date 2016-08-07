#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#include <iostream>
#include <string>
#include <vector>
#include <fstream>

using namespace std;

#define MAX_STRING 100
#define EXP_TABLE_SIZE 512
#define LOG_TABLE_SIZE 512
#define MAX_EXP 8
#define MAX_SENTENCE_LENGTH 1000
#define MAX_CODE_LENGTH 40

typedef float real;

char test_file[MAX_STRING], model_file[MAX_STRING];

const int vocab_hash_size = 30000000;  // Maximum 30 * 0.7 = 21M words in the vocabulary.

int total_nodes = 0;

struct node {
    char *name;
    int id;
    struct node *left;
    struct node *right;
};
struct node* huffman_tree;

struct Sentence {
    vector<string> words_;
    string label_;
};

struct vocab_word {
    long long cn; // Frequency of word.
    char *word;  // word string.
};

int *vocab_hash; // Hash table.
struct vocab_word *vocab;
long long vocab_size = 0, category_size = 0, layer1_size = 10;
real *syn0, *syn1, *expTable, *logTable;

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

int ArgPos(const char *str, int argc, char **argv) {
    for (int a = 1; a < argc; a++) {
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

real getLog(real x) {
    if (x > 1.0 ) return 0.0;
    return logTable[(int) (x * LOG_TABLE_SIZE)];
}

void BuildTree(const string &label, const string &code, const string &point) {
    vector<string> points;
    Split(point, ",", points);
    if (code.size() != points.size()) {
        cerr << "error, size not match" << endl;
        return;
    }

    struct node *cursor = huffman_tree;
    struct node *previous = cursor;
    for (size_t i = 0; i < code.size(); i++) {
        previous = cursor;
        if (cursor->id == -1) {
            cursor->id = atoi(points[i].c_str());
        }
        if (code[i] == '0') {
            cursor = cursor->left;
        } else {
            cursor = cursor->right;
        }

        if (cursor == NULL) {
            cursor = (struct node *) malloc(sizeof(struct node));
            cursor->id = -1;
            cursor->left = NULL;
            cursor->right = NULL;
            total_nodes++;
        }

        if (code[i] == '0') {
            previous->left = cursor;
        } else {
            previous->right = cursor;
        }
    }
    cursor->left = NULL;
    cursor->right = NULL;
    cursor->name = (char *) malloc(sizeof(char) * MAX_STRING);
    strcpy(cursor->name, label.c_str());
}

// DFS Destory?
void DestroyTree(void) {
}


// pre-order visist
void PreOrder(struct node *sub, vector<int> &code, vector<int> &parent) {
    if (sub != NULL) {
        if (sub->left == NULL && sub->right == NULL) {
            for (size_t i = 0; i < code.size(); i++) {
                fprintf(stderr, "%d", code[i]);
            }
            fprintf(stderr, " ");
            for (size_t i = 0; i < parent.size(); i++) {
                fprintf(stderr, "%d,", parent[i]);
            }
            fprintf(stderr, ":%s\n", sub->name);
        }
        if (sub->left != NULL) {
            code.push_back(0);
            parent.push_back(sub->id);
            PreOrder(sub->left, code, parent);
            code.pop_back();
            parent.pop_back();
        }
        if (sub->right != NULL) {
            code.push_back(1);
            parent.push_back(sub->id);
            PreOrder(sub->right, code, parent);
            code.pop_back();
            parent.pop_back();
        }
    }
}

void PreOrderCompute(struct node *sub, real *neu1, real &max_prob, real score, char *label) {
    if (score < max_prob) return;
    if (sub->left == NULL && sub->right == NULL) {  // leaf node.
        max_prob = score;
        strcpy(label, sub->name);
        return;
    }

    real f = 0.0;
    for (long long a = 0; a < layer1_size; a++) {
        f += neu1[a] * syn1[a + sub->id * layer1_size];
    }

    if (f >= MAX_EXP) {
        f = 1.0;
    } else if (f <= -MAX_EXP) {
        f = 0.0;
    } else {
        f = expTable[(int) ((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
    }
    
    PreOrderCompute(sub->left, neu1, max_prob, score + getLog(1 - f), label);
    PreOrderCompute(sub->right, neu1, max_prob, score + getLog(f), label);
}

// Returns hash value of a word
int GetWordHash(string word) {
    unsigned long long a, hash = 0;
    for (a = 0; a < word.size(); a++) {
        hash = hash * 257 + word[a];
    }
    hash %= vocab_hash_size;
    return hash;
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

void Predict(Sentence *sen, float &prob) {
    long long wid;
    long long a;
    real *neu1 = (real *) calloc(layer1_size, sizeof(real));
    for (a = 0; a < layer1_size; a++) neu1[a] = 0;

    int total = 0;
    for (size_t i = 1; i < sen->words_.size(); ++i) {
        wid = SearchVocab(sen->words_[i].c_str());
        if (wid == -1) continue;
        total++;
        for (a = 0; a < layer1_size; a++) neu1[a] += syn0[a + wid * layer1_size];
    }
    for (a = 0; a < layer1_size; a++) neu1[a] /= total;
    
    real pmax = -1e10;
    char label[MAX_STRING];
    real score = 0.0;
    PreOrderCompute(huffman_tree, neu1, pmax, score, label);
    prob = pmax;
    sen->label_ = string(label);
}

int AddWordToVocab(const char *word, int idx) {
    unsigned int hash, length = strlen(word) + 1;
    if (length > MAX_STRING) length = MAX_STRING;
    vocab[idx].word = (char *) calloc(length, sizeof(char));
    strcpy(vocab[idx].word, word);
    hash = GetWordHash(word);
    while(vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
    vocab_hash[hash] = idx;
    return idx;
}

void LoadModel() {
    long long a, c;
    ifstream fin(model_file);
    if (!fin.is_open()) {
        fprintf(stderr, "open file [%s] failed.\n", model_file);
        exit(1);
    }
    string line;
    getline(fin, line);
    sscanf(line.c_str(), "%lld %lld %lld", &vocab_size, &category_size, &layer1_size);
    vocab = (struct vocab_word *) calloc(vocab_size, sizeof(struct vocab_word));
    vocab_hash = (int *) calloc(vocab_hash_size, sizeof(int));
    for (c = 0; c < vocab_hash_size; c++) vocab_hash[c] = -1;
    a = posix_memalign((void **) &syn0, 128,
                    vocab_size * layer1_size * sizeof(real));

    a = posix_memalign((void **) &syn1, 128,
                    category_size * layer1_size * sizeof(real));
    // Init vocab table.
    fprintf(stderr, "loading vocabulary vectors...\n");
    vector<string> token;
    for (size_t i = 0; i < vocab_size; i++) {
        getline(fin, line);
        token.clear();
        Split(line, " ", token);
        AddWordToVocab(token[0].c_str(), i);
        for (c = 0; c < layer1_size; c++) {
            syn0[i * layer1_size + c] = atof(token[c+1].c_str());
        }
    }
    fprintf(stderr, "loading virtual category vectors...\n");
    // category weights.
    for (size_t i = 0; i < category_size; i++) {
        getline(fin, line);
        token.clear();
        Split(line, " ", token);
        for (c = 0; c < layer1_size; c++) {
            syn1[i * layer1_size + c] = atof(token[c].c_str());
        }
    }
    fprintf(stderr, "loading category code & point...\n");
    // category huffman code.
    for (size_t i = 0; i < category_size; i++) {
        getline(fin, line);
        int idx1 = line.find(' ');
        string label = line.substr(0, idx1);
        int idx2 = line.find(' ', idx1+1);
        string code = line.substr(idx1+1, idx2-idx1 - 1);
        string point = line.substr(idx2+1);
        BuildTree(label, code, point);
    }
    fin.close();
    fprintf(stderr, "complete.\n");
}

int main(int argc, char *argv[]) {
    int i;
    if ((i = ArgPos("-test", argc, argv)) > 0) strcpy(test_file, argv[i + 1]);
    if ((i = ArgPos("-model", argc, argv)) > 0) strcpy(model_file, argv[i + 1]);

    expTable = (real *) malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
    if (expTable == NULL) {
        fprintf(stderr, "out of memory\n");
        exit(1);
    }
    for (i = 0; i < EXP_TABLE_SIZE; i++) {
        real x = real(i * 2 * MAX_EXP) / EXP_TABLE_SIZE - MAX_EXP;
        expTable[i] = 1.0 / (1.0 + exp(-x));
    }
    logTable = (real *) malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
    for (i = 0; i < LOG_TABLE_SIZE; i++) {
        real x = (real(i) + 1e-5) / LOG_TABLE_SIZE;
        logTable[i] = log(x);
    }

    total_nodes = 1;

    huffman_tree = (struct node *) malloc(sizeof(struct node));
    huffman_tree->name = (char *) malloc(sizeof(char) * MAX_STRING);
    strcpy(huffman_tree->name, "root");
    huffman_tree->left = NULL;
    huffman_tree->right = NULL;
    huffman_tree->id = -1;

    LoadModel();

    /*vector<int> code;
    vector<int> parent;
    PreOrder(huffman_tree, code, parent);*/
    ifstream fin(test_file);
    if (!fin.is_open()) {
        cerr << "open " << test_file << " failed." << endl;
        return -1;
    }

    string line;
    float prob;
    while (getline(fin, line)) {
        Sentence *sen = new Sentence();
        Split(line, " ", sen->words_);
        Predict(sen, prob);
        cout << sen->label_ << endl;
        delete sen;
    }

    fin.close();

    free(expTable);
    free(logTable);
    return 0;
}

