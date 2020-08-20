/* Copyright 2020 The TensorFlow Quantum Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "tensorflow_quantum/core/src/adj_util.h"

#include <functional>
#include <string>
#include <vector>

#include "../qsim/lib/circuit.h"
#include "../qsim/lib/fuser.h"
#include "../qsim/lib/fuser_basic.h"
#include "../qsim/lib/gates_cirq.h"
#include "../qsim/lib/io.h"
#include "../qsim/lib/matrix.h"

namespace tfq {

static const float _GRAD_EPS = 5e-3;

typedef qsim::Cirq::GateCirq<float> QsimGate;

void PopulateGradientSingleEigen(
    const std::function<QsimGate(unsigned int, unsigned int, float, float)>&
        create_f,
    const std::string& symbol, unsigned int location, unsigned int qid,
    float exp, float exp_s, float gs, GradientOfGate* grad) {
  grad->params.push_back(symbol);
  grad->index = location;
  auto left = create_f(0, qid, (exp + _GRAD_EPS) * exp_s, gs);
  auto right = create_f(0, qid, (exp - _GRAD_EPS) * exp_s, gs);
  Matrix2Diff(right.matrix,
              left.matrix);  // left's entries have right subtracted.
  qsim::Matrix2ScalarMultiply(0.5 / _GRAD_EPS, left.matrix);
  grad->grad_gates.push_back(left);
}

void PopulateGradientTwoEigen(
    const std::function<QsimGate(unsigned int, unsigned int, unsigned int,
                                 float, float)>& create_f,
    const std::string& symbol, unsigned int location, unsigned int qid,
    unsigned int qid2, float exp, float exp_s, float gs, GradientOfGate* grad) {
  grad->params.push_back(symbol);
  grad->index = location;
  auto left = create_f(0, qid, qid2, (exp + _GRAD_EPS) * exp_s, gs);
  auto right = create_f(0, qid, qid2, (exp - _GRAD_EPS) * exp_s, gs);
  Matrix4Diff(right.matrix,
              left.matrix);  // left's entries have right subtracted.
  qsim::Matrix4ScalarMultiply(0.5 / _GRAD_EPS, left.matrix);
  grad->grad_gates.push_back(left);
}

void PopulateGradientPhasedXPhasedExponent(const std::string& symbol,
                                           unsigned int location,
                                           unsigned int qid, float pexp,
                                           float pexp_s, float exp, float exp_s,
                                           float gs, GradientOfGate* grad) {
  grad->params.push_back(symbol);
  grad->index = location;
  auto left = qsim::Cirq::PhasedXPowGate<float>::Create(
      0, qid, (pexp + _GRAD_EPS) * pexp_s, exp * exp_s, gs);
  auto right = qsim::Cirq::PhasedXPowGate<float>::Create(
      0, qid, (pexp - _GRAD_EPS) * pexp_s, exp * exp_s, gs);
  Matrix2Diff(right.matrix,
              left.matrix);  // left's entries have right subtracted.
  qsim::Matrix2ScalarMultiply(0.5 / _GRAD_EPS, left.matrix);
  grad->grad_gates.push_back(left);
}

void PopulateGradientPhasedXExponent(const std::string& symbol,
                                     unsigned int location, unsigned int qid,
                                     float pexp, float pexp_s, float exp,
                                     float exp_s, float gs,
                                     GradientOfGate* grad) {
  grad->params.push_back(symbol);
  grad->index = location;
  auto left = qsim::Cirq::PhasedXPowGate<float>::Create(
      0, qid, pexp * pexp_s, (exp + _GRAD_EPS) * exp_s, gs);
  auto right = qsim::Cirq::PhasedXPowGate<float>::Create(
      0, qid, pexp * pexp_s, (exp - _GRAD_EPS) * exp_s, gs);
  Matrix2Diff(right.matrix,
              left.matrix);  // left's entries have right subtracted.
  qsim::Matrix2ScalarMultiply(0.5 / _GRAD_EPS, left.matrix);
  grad->grad_gates.push_back(left);
}

void PopulateGradientFsimTheta(const std::string& symbol, unsigned int location,
                               unsigned int qid, unsigned qid2, float theta,
                               float theta_s, float phi, float phi_s,
                               GradientOfGate* grad) {
  grad->params.push_back(symbol);
  grad->index = location;
  auto left = qsim::Cirq::FSimGate<float>::Create(
      0, qid, qid2, (theta + _GRAD_EPS) * theta_s, phi * phi_s);
  auto right = qsim::Cirq::FSimGate<float>::Create(
      0, qid, qid2, (theta - _GRAD_EPS) * theta_s, phi * phi_s);
  Matrix4Diff(right.matrix,
              left.matrix);  // left's entries have right subtracted.
  qsim::Matrix4ScalarMultiply(0.5 / _GRAD_EPS, left.matrix);
  grad->grad_gates.push_back(left);
}

void PopulateGradientFsimPhi(const std::string& symbol, unsigned int location,
                             unsigned int qid, unsigned qid2, float theta,
                             float theta_s, float phi, float phi_s,
                             GradientOfGate* grad) {
  grad->params.push_back(symbol);
  grad->index = location;
  auto left = qsim::Cirq::FSimGate<float>::Create(0, qid, qid2, theta * theta_s,
                                                  (phi + _GRAD_EPS) * phi_s);
  auto right = qsim::Cirq::FSimGate<float>::Create(
      0, qid, qid2, theta * theta_s, (phi - _GRAD_EPS) * phi_s);
  Matrix4Diff(right.matrix,
              left.matrix);  // left's entries have right subtracted.
  qsim::Matrix4ScalarMultiply(0.5 / _GRAD_EPS, left.matrix);
  grad->grad_gates.push_back(left);
}

void PopulateGradientPhasedISwapPhasedExponent(
    const std::string& symbol, unsigned int location, unsigned int qid,
    unsigned int qid2, float pexp, float pexp_s, float exp, float exp_s,
    GradientOfGate* grad) {
  grad->params.push_back(symbol);
  grad->index = location;
  auto left = qsim::Cirq::PhasedISwapPowGate<float>::Create(
      0, qid, qid2, (pexp + _GRAD_EPS) * pexp_s, exp * exp_s);
  auto right = qsim::Cirq::PhasedISwapPowGate<float>::Create(
      0, qid, qid2, (pexp - _GRAD_EPS) * pexp_s, exp * exp_s);
  Matrix4Diff(right.matrix,
              left.matrix);  // left's entries have right subtracted.
  qsim::Matrix4ScalarMultiply(0.5 / _GRAD_EPS, left.matrix);
  grad->grad_gates.push_back(left);
}

void PopulateGradientPhasedISwapExponent(const std::string& symbol,
                                         unsigned int location,
                                         unsigned int qid, unsigned int qid2,
                                         float pexp, float pexp_s, float exp,
                                         float exp_s, GradientOfGate* grad) {
  grad->params.push_back(symbol);
  grad->index = location;
  auto left = qsim::Cirq::PhasedISwapPowGate<float>::Create(
      0, qid, qid2, pexp * pexp_s, (exp + _GRAD_EPS) * exp_s);
  auto right = qsim::Cirq::PhasedISwapPowGate<float>::Create(
      0, qid, qid2, pexp * pexp_s, (exp - _GRAD_EPS) * exp_s);
  Matrix4Diff(right.matrix,
              left.matrix);  // left's entries have right subtracted.
  qsim::Matrix4ScalarMultiply(0.5 / _GRAD_EPS, left.matrix);
  grad->grad_gates.push_back(left);
}

}  // namespace tfq
