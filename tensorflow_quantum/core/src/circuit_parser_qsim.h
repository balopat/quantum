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

#ifndef TFQ_CORE_SRC_CIRCUIT_PARSER_QSIM_H_
#define TFQ_CORE_SRC_CIRCUIT_PARSER_QSIM_H_

#include <string>
#include <vector>

#include "../qsim/lib/circuit.h"
#include "../qsim/lib/fuser.h"
#include "../qsim/lib/gates_cirq.h"
#include "absl/container/flat_hash_map.h"
#include "cirq/google/api/v2/program.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow_quantum/core/proto/pauli_sum.pb.h"
#include "tensorflow_quantum/core/src/adj_util.h"

namespace tfq {

// parse a serialized Cirq program into a qsim representation.
// ingests a Cirq Circuit proto and produces a resolved qsim Circuit,
// as well as a fused circuit.
tensorflow::Status QsimCircuitFromProgram(
    const cirq::google::api::v2::Program& program,
    const absl::flat_hash_map<std::string, std::pair<int, float>>& param_map,
    const int num_qubits, qsim::Circuit<qsim::Cirq::GateCirq<float>>* circuit,
    std::vector<qsim::GateFused<qsim::Cirq::GateCirq<float>>>* fused_circuit);

// parse a serialized Cirq program into a qsim representation.
// ingests a Cirq Circuit proto and produces a resolved qsim Circuit.
// will also track gates containing symbols and provide metadata in
// grad_indices. fusing of the circuit will also block on gates with symbols
// found in them. This functionality is particularly useful for the adjoint
// differentiation algorithm. assumes at least one symbol exists in given
// circuit.
tensorflow::Status QsimCircuitFromProgramADJ(
    const cirq::google::api::v2::Program& program,
    const absl::flat_hash_map<std::string, std::pair<int, float>>& param_map,
    const int num_qubits, qsim::Circuit<qsim::Cirq::GateCirq<float>>* circuit,
    std::vector<std::vector<qsim::GateFused<qsim::Cirq::GateCirq<float>>>>*
        partial_fuses,
    std::vector<GradientOfGate>* grad_indices);

// parse a serialized pauliTerm from a larger cirq.Paulisum proto
// into a qsim Circuit and fused circuit.
tensorflow::Status QsimCircuitFromPauliTerm(
    const tfq::proto::PauliTerm& term, const int num_qubits,
    qsim::Circuit<qsim::Cirq::GateCirq<float>>* circuit,
    std::vector<qsim::GateFused<qsim::Cirq::GateCirq<float>>>* fused_circuit);

// parse a serialized pauliTerm from a larger cirq.Paulisum proto
// into a qsim Circuit and fused circuit that represents the transformation
// to the z basis.
tensorflow::Status QsimZBasisCircuitFromPauliTerm(
    const tfq::proto::PauliTerm& term, const int num_qubits,
    qsim::Circuit<qsim::Cirq::GateCirq<float>>* circuit,
    std::vector<qsim::GateFused<qsim::Cirq::GateCirq<float>>>* fused_circuit);

}  // namespace tfq

#endif  // TFQ_CORE_SRC_CIRCUIT_PARSER_QSIM_H_
