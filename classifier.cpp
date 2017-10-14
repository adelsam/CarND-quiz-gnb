#include <iostream>
#include <sstream>
#include <fstream>
#include <math.h>
#include <vector>
#include "classifier.h"

/**
 * Initializes GNB
 */
static double calc_mean(vector<double> values) {
  double total = 0.0;
  for (double v : values) {
    total += v;
  }
  return total / values.size();
}

static double calc_stdd(double mean, vector<double> values) {
  double total = 0.0;
  for (double v : values) {
    total += pow((v - mean), 2);
  }
  return sqrt(total / values.size());
}


GNB::GNB() {

}

GNB::~GNB() {}

long GNB::label_index(string value) {
  return std::distance(possible_labels.begin(),
                       std::find(possible_labels.begin(), possible_labels.end(), value));
}

void GNB::train(vector<vector<double>> data, vector<string> labels) {

  /*
    Trains the classifier with N data points and labels.

    INPUTS
    data - array of N observations
      - Each observation is a tuple with 4 values: s, d,
        s_dot and d_dot.
      - Example : [
          [3.5, 0.1, 5.9, -0.02],
          [8.0, -0.3, 3.0, 2.2],
          ...
        ]

    labels - array of N labels
      - Each label is one of "left", "keep", or "right".
  */
  int num_vars = 4;
  // [label][j][...]
  vector<vector<vector<double>>> totals(possible_labels.size(),
                                        vector<vector<double>>(num_vars, vector<double>()));

  for (int i = 0; i < data.size(); i++) {
    long idx = label_index(labels[i]);
    for (int j = 0; j < num_vars; j++) {
      totals[idx][j].push_back(data[i][j]);
    }
  }

  // [label][j]
  mean = vector<vector<double>>(possible_labels.size(), vector<double>(num_vars));
  stdd = vector<vector<double>>(possible_labels.size(), vector<double>(num_vars));

  for (int i = 0; i < possible_labels.size(); i++) {
    for (int j = 0; j < num_vars; j++) {
      vector<double> values = totals[i][j];
      double m = calc_mean(values);
      double s = calc_stdd(m, values);
      cout << "i = " << i << " j = " << j << " mean: " << m
           << " stdd: " << s << " n = " << values.size() << endl;
      mean[i][j] = m;
      stdd[i][j] = s;
    }
  }

}

string GNB::predict(vector<double> sample) {
  /*
    Once trained, this method is called and expected to return
    a predicted behavior for the given observation.

    INPUTS

    observation - a 4 tuple with s, d, s_dot, d_dot.
      - Example: [3.5, 0.1, 8.5, -0.2]

    OUTPUT

    A label representing the best guess of the classifier. Can
    be one of "left", "keep" or "right".
    """
    # TODO - complete this
  */

  return this->possible_labels[1];

}