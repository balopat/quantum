# Copyright 2020 The TensorFlow Quantum Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests that specifically target tfq_unitary_op."""
import numpy as np
from absl.testing import parameterized
import tensorflow as tf
import cirq
import sympy

from tensorflow_quantum.python import util
from tensorflow_quantum.core.ops import tfq_adj_grad_op


class ADJGradTest(tf.test.TestCase, parameterized.TestCase):
    """Tests tfq_calculate_unitary."""

    def test_calculate_adj_grad_simple_case(self):
        """Make sure that adjoint gradient works on simple input case."""
        n_qubits = 2
        batch_size = 1
        symbol_names = ['alpha', 'beta']
        qubits = cirq.GridQubit.rect(1, n_qubits)
        circuit_batch, resolver_batch = \
        [cirq.Circuit(cirq.X(qubits[0]) ** sympy.Symbol('alpha'),
            cirq.Y(qubits[1]) ** sympy.Symbol('beta'),
            cirq.CNOT(qubits[0], qubits[1]))], [{'alpha': 0.123, 'beta':0.456}]

        op_batch = [[cirq.Z(qubits[0]), cirq.X(qubits[1])] for _ in range(batch_size)]

        symbol_values_array = np.array(
            [[resolver[symbol]
              for symbol in symbol_names]
             for resolver in resolver_batch])

        prev_grads = tf.ones([batch_size, len(symbol_names)])

        out = tfq_adj_grad_op.tfq_adj_grad(
            util.convert_to_tensor(circuit_batch),
            tf.convert_to_tensor(symbol_names),
            tf.convert_to_tensor(symbol_values_array),
            util.convert_to_tensor(op_batch),
            prev_grads)

        self.assertAllClose(out,
            np.array([[-1.18392, 0.43281]]),
            atol=1e-3)

    def test_calculate_adj_grad_simple_case2(self):
        """Make sure the adjoint gradient works on another simple input case."""
        n_qubits = 2
        batch_size = 1
        symbol_names = ['alpha', 'beta', 'gamma']
        qubits = cirq.GridQubit.rect(1, n_qubits)
        circuit_batch, resolver_batch = \
        [cirq.Circuit(cirq.X(qubits[0]) ** sympy.Symbol('alpha'),
            cirq.Y(qubits[1]) ** sympy.Symbol('beta'),
            cirq.CNOT(qubits[0], qubits[1]),
            cirq.FSimGate(sympy.Symbol('gamma'), 0.5)(qubits[0], qubits[1]))
        ], [{'alpha': 0.123, 'beta':0.456, 'gamma':0.789}]

        op_batch = [[cirq.Z(qubits[0]), cirq.X(qubits[1])] for _ in range(batch_size)]

        symbol_values_array = np.array(
            [[resolver[symbol]
              for symbol in symbol_names]
             for resolver in resolver_batch])

        prev_grads = tf.ones([batch_size, len(op_batch[0])])

        out = tfq_adj_grad_op.tfq_adj_grad(
            util.convert_to_tensor(circuit_batch),
            tf.convert_to_tensor(symbol_names),
            tf.convert_to_tensor(symbol_values_array),
            util.convert_to_tensor(op_batch),
            prev_grads)

        print(out)
        self.assertAllClose(out,
                            np.array([[-2.100, -1.7412, -1.5120]]),
                            atol=1e-3)

    def test_calculate_adj_grad_simple_case3(self):
        """Make sure the adjoint gradient works on ANOTHER simple input case."""
        n_qubits = 2
        batch_size = 1
        symbol_names = ['alpha', 'beta', 'gamma']
        qubits = cirq.GridQubit.rect(1, n_qubits)
        circuit_batch, resolver_batch = \
        [cirq.Circuit(cirq.X(qubits[0]) ** sympy.Symbol('alpha'),
            cirq.Y(qubits[1]) ** sympy.Symbol('beta'),
            cirq.CNOT(qubits[0], qubits[1]),
            cirq.FSimGate(sympy.Symbol('gamma'), sympy.Symbol('gamma'))(qubits[0], qubits[1]))
        ], [{'alpha': 0.123, 'beta':0.456, 'gamma':0.789}]

        op_batch = [[cirq.Z(qubits[0]), cirq.X(qubits[1])] for _ in range(batch_size)]

        symbol_values_array = np.array(
            [[resolver[symbol]
              for symbol in symbol_names]
             for resolver in resolver_batch])

        prev_grads = tf.ones([batch_size, len(op_batch[0])])

        out = tfq_adj_grad_op.tfq_adj_grad(
            util.convert_to_tensor(circuit_batch),
            tf.convert_to_tensor(symbol_names),
            tf.convert_to_tensor(symbol_values_array),
            util.convert_to_tensor(op_batch),
            prev_grads)

        print(out)
        self.assertAllClose(out,
                            np.array([[-2.3484, -1.7532, -1.64264]]),
                            atol=1e-3)


if __name__ == "__main__":
    tf.test.main()
