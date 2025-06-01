#include <iostream>
#include <sstream>
#include <vector>
#include <map>
#include <set>
#include <cmath>
#include <algorithm>

using namespace std;

vector<string> tokenize(const string& message) {
    istringstream stream(message);
    string word;
    vector<string> tokens;
    while (stream >> word) {
        transform(word.begin(), word.end(), word.begin(), ::tolower);
        tokens.push_back(word);
    }
    return tokens;
}

class NaiveBayesClassifier {
    map<string, map<string, int>> wordCounts;
    map<string, int> classCounts;
    map<string, int> totalWords;
    set<string> vocabulary;

public:
    void train(const vector<pair<string, string>>& dataset) {
        for (const auto& entry : dataset) {
            string message = entry.first;
            string label = entry.second;
            vector<string> tokens = tokenize(message);

            classCounts[label]++;
            for (const string& token : tokens) {
                wordCounts[label][token]++;
                totalWords[label]++;
                vocabulary.insert(token);
            }
        }
    }

    string predict(const string& message) {
        vector<string> tokens = tokenize(message);
        int totalMessages = 0;
        for (const auto& c : classCounts) totalMessages += c.second;

        map<string, double> classScores;
        for (const auto& label : classCounts) {
            double logProb = log((double)label.second / totalMessages);
            for (const string& token : tokens) {
                int count = wordCounts[label.first][token];
                double prob = (count + 1.0) / (totalWords[label.first] + vocabulary.size());
                logProb += log(prob);
            }
            classScores[label.first] = logProb;
        }

        return max_element(classScores.begin(), classScores.end(),
            [](const pair<string, double>& a, const pair<string, double>& b) {
                return a.second < b.second;
            })->first;
    }
};

int main() {
    vector<pair<string, string>> dataset = {
        {"win money now", "spam"}, {"limited offer", "spam"}, {"team meeting", "not_spam"},
        {"project deadline", "not_spam"}, {"click here for cash", "spam"}, {"happy birthday", "not_spam"}
        // Add more for complete 50/50 dataset
    };

    NaiveBayesClassifier classifier;
    classifier.train(dataset);

    string testMessage = "win cash now";
    cout << "Prediction: " << classifier.predict(testMessage) << endl;
    return 0;
}
