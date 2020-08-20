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

#include <string>
#include <vector>

#include "../qsim/lib/circuit.h"
#include "../qsim/lib/fuser.h"
#include "../qsim/lib/fuser_basic.h"
#include "../qsim/lib/gates_cirq.h"
#include "../qsim/lib/io.h"
#include "../qsim/lib/matrix.h"
#include "gtest/gtest.h"

namespace tfq {
namespace {

TEST(AdjUtilTest, SingleEigenGrad) {
  GradientOfGate grad;

  PopulateGradientSingleEigen(&qsim::Cirq::YPowGate<float>::Create, "hello", 5,
                              2, 0.125, 1.0, 0.0, &grad);

  // Value verified from:
  /*
  (cirq.unitary(cirq.Y**(0.125 + 1e-4)) -
   cirq.unitary(cirq.Y**(0.125 - 1e-4))) / 2e-4
  array([[-0.60111772+1.45122655j, -1.45122655-0.60111772j],
         [ 1.45122655+0.60111772j, -0.60111772+1.45122655j]])
  */

  EXPECT_EQ(grad.index, 5);
  EXPECT_EQ(grad.params[0], "hello");
  EXPECT_NEAR(grad.grad_gates[0].matrix[0], -0.60111, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[1], 1.45122, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[2], -1.45122, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[3], -0.60111, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[4], 1.45122, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[5], 0.60111, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[6], -0.60111, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[7], 1.45122, 1e-4);
}

TEST(AdjUtilTest, TwoEigenGrad) {
  GradientOfGate grad;

  PopulateGradientTwoEigen(&qsim::Cirq::XXPowGate<float>::Create, "hi", 5, 2, 3,
                           0.001, 1.0, 0.0, &grad);

  // Value verified from:
  /*
  (cirq.unitary(cirq.XX**(0.001 + 1e-4)) -
   cirq.unitary(cirq.XX**(0.001 - 1e-4))) / 2e-4
    array([[-0.00493479+1.57078855j,  0.        +0.j        ,
             0.        +0.j        ,  0.00493479-1.57078855j],
           [ 0.        +0.j        , -0.00493479+1.57078855j,
             0.00493479-1.57078855j,  0.        +0.j        ],
           [ 0.        +0.j        ,  0.00493479-1.57078855j,
            -0.00493479+1.57078855j,  0.        +0.j        ],
           [ 0.00493479-1.57078855j,  0.        +0.j        ,
             0.        +0.j        , -0.00493479+1.57078855j]])
  */

  EXPECT_EQ(grad.index, 5);
  EXPECT_EQ(grad.params[0], "hi");
  EXPECT_NEAR(grad.grad_gates[0].matrix[0], -0.004934, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[1], 1.57078, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[2], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[3], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[4], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[5], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[6], 0.004934, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[7], -1.57078, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[8], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[9], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[10], -0.004934, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[11], 1.57078, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[12], 0.004934, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[13], -1.57078, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[14], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[15], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[16], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[17], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[18], 0.004934, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[19], -1.57078, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[20], -0.004934, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[21], 1.57078, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[22], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[23], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[24], 0.004934, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[25], -1.57078, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[26], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[27], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[28], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[29], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[30], -0.004934, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[31], 1.57078, 1e-4);
}

TEST(AdjUtilTest, PhasedXPhasedExponent) {
  GradientOfGate grad;

  PopulateGradientPhasedXPhasedExponent("hello2", 5, 2, 10.123, 1.0, 1.0, 1.0,
                                        0.0, &grad);

  /*
  (cirq.unitary(cirq.PhasedXPowGate(exponent=1.0,phase_exponent=0.001 + 1e-4)) -
   cirq.unitary(cirq.PhasedXPowGate(exponent=1.0,phase_exponent=0.001 - 1e-4)))
     / 2e-4
    array([[ 0.        +0.j        , -1.18397518-2.90994963j],
           [-1.18397518+2.90994963j,  0.        +0.j        ]])

  */
  EXPECT_EQ(grad.index, 5);
  EXPECT_EQ(grad.params[0], "hello2");
  EXPECT_NEAR(grad.grad_gates[0].matrix[0], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[1], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[2], -1.18397, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[3], -2.9099, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[4], -1.18397, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[5], 2.9099, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[6], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[7], 0.0, 1e-4);
}

TEST(AdjUtilTest, PhasedXExponent) {
  GradientOfGate grad;

  PopulateGradientPhasedXExponent("hello3", 5, 2, 10.123, 1.0, 0.789, 1.0, 0.0,
                                  &grad);
  /*
  (cirq.unitary(cirq.PhasedXPowGate(exponent=0.789+1e-4,phase_exponent=10.123))
  -
  cirq.unitary(cirq.PhasedXPowGate(exponent=0.789-1e-4,phase_exponent=10.123)))
  / 2e-4 array([[-0.96664663-1.23814188j,  1.36199145+0.78254732j], [
  0.42875189+1.51114951j, -0.96664663-1.23814188j]])
  */

  EXPECT_EQ(grad.index, 5);
  EXPECT_EQ(grad.params[0], "hello3");
  EXPECT_NEAR(grad.grad_gates[0].matrix[0], -0.96664, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[1], -1.23814, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[2], 1.36199, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[3], 0.78254, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[4], 0.42875, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[5], 1.51114, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[6], -0.96664, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[7], -1.23814, 1e-4);
}

TEST(AdjUtilTest, FSimThetaGrad) {
  GradientOfGate grad;
  PopulateGradientFsimTheta("hihi", 5, 2, 3, 0.5, 1.0, 1.2, 1.0, &grad);

  /*
  (cirq.unitary(cirq.FSimGate(theta=0.5 + 1e-4,phi=1.2)) -
   cirq.unitary(cirq.FSimGate(theta=0.5-1e-4,phi=1.2))) / 2e-4
    array([[ 0.        +0.j        ,  0.        +0.j        ,
             0.        +0.j        ,  0.        +0.j        ],
           [ 0.        +0.j        , -0.47942554+0.j        ,
             0.        -0.87758256j,  0.        +0.j        ],
           [ 0.        +0.j        ,  0.        -0.87758256j,
            -0.47942554+0.j        ,  0.        +0.j        ],
           [ 0.        +0.j        ,  0.        +0.j        ,
             0.        +0.j        ,  0.        +0.j        ]])
  */

  EXPECT_EQ(grad.index, 5);
  EXPECT_EQ(grad.params[0], "hihi");
  EXPECT_NEAR(grad.grad_gates[0].matrix[0], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[1], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[2], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[3], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[4], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[5], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[6], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[7], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[8], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[9], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[10], -0.47942, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[11], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[12], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[13], -0.87758, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[14], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[15], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[16], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[17], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[18], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[19], -0.87758, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[20], -0.47942, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[21], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[22], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[23], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[24], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[25], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[26], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[27], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[28], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[29], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[30], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[31], 0.0, 1e-4);
}

TEST(AdjUtilTest, FSimPhiGrad) {
  GradientOfGate grad;
  PopulateGradientFsimPhi("hihi2", 5, 2, 3, 0.5, 1.0, 1.2, 1.0, &grad);

  /*
  (cirq.unitary(cirq.FSimGate(theta=0.5,phi=1.2+1e-4)) -
  cirq.unitary(cirq.FSimGate(theta=0.5,phi=1.2-1e-4))) / 2e-4
  array([[ 0.        +0.j        ,  0.        +0.j        ,
           0.        +0.j        ,  0.        +0.j        ],
         [ 0.        +0.j        ,  0.        +0.j        ,
           0.        +0.j        ,  0.        +0.j        ],
         [ 0.        +0.j        ,  0.        +0.j        ,
           0.        +0.j        ,  0.        +0.j        ],
         [ 0.        +0.j        ,  0.        +0.j        ,
           0.        +0.j        , -0.93203908-0.36235775j]])
  */

  EXPECT_EQ(grad.index, 5);
  EXPECT_EQ(grad.params[0], "hihi2");
  EXPECT_NEAR(grad.grad_gates[0].matrix[0], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[1], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[2], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[3], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[4], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[5], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[6], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[7], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[8], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[9], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[10], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[11], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[12], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[13], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[14], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[15], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[16], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[17], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[18], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[19], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[20], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[21], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[22], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[23], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[24], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[25], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[26], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[27], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[28], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[29], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[30], -0.932039, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[31], -0.362357, 1e-4);
}

TEST(AdjUtilTest, PhasedISwapPhasedExponent) {
  GradientOfGate grad;

  PopulateGradientPhasedISwapPhasedExponent("h", 5, 3, 2, 8.9, 1.0, -3.2, 1.0,
                                            &grad);

  /*
  (cirq.unitary(cirq.PhasedISwapPowGate(exponent=-3.2,phase_exponent=8.9+1e-4))
  - cirq.unitary(cirq.PhasedISwapPowGate(exponent=3.2,phase_exponent=8.9-1e-4)))
    / 2e-4
  array([[ 0.        +0.j        ,  0.        +0.j        ,
           0.        +0.j        ,  0.        +0.j        ],
         [ 0.        +0.j        ,  0.        +0.j        ,
          -4.83441368+3.51240713j,  0.        +0.j        ],
         [ 0.        +0.j        ,  4.83441368+3.51240713j,
           0.        +0.j        ,  0.        +0.j        ],
         [ 0.        +0.j        ,  0.        +0.j        ,
           0.        +0.j        ,  0.        +0.j        ]])

  */

  EXPECT_EQ(grad.index, 5);
  EXPECT_EQ(grad.params[0], "h");
  EXPECT_NEAR(grad.grad_gates[0].matrix[0], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[1], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[2], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[3], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[4], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[5], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[6], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[7], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[8], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[9], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[10], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[11], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[12], -4.83441, 1e-3);
  EXPECT_NEAR(grad.grad_gates[0].matrix[13], 3.51238, 1e-3);
  EXPECT_NEAR(grad.grad_gates[0].matrix[14], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[15], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[16], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[17], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[18], 4.83441, 1e-3);
  EXPECT_NEAR(grad.grad_gates[0].matrix[19], 3.51238, 1e-3);
  EXPECT_NEAR(grad.grad_gates[0].matrix[20], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[21], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[22], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[23], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[24], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[25], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[26], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[27], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[28], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[29], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[30], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[31], 0.0, 1e-4);
}

TEST(AdjUtilTest, PhasedISwapExponent) {
  GradientOfGate grad;

  PopulateGradientPhasedISwapExponent("h2", 5, 3, 2, 8.9, 1.0, -3.2, 1.0,
                                      &grad);

  /*
  (cirq.unitary(cirq.PhasedISwapPowGate(exponent=-3.2+1e-3,phase_exponent=8.9))
  -cirq.unitary(cirq.PhasedISwapPowGate(exponent=-3.2-1e-3,phase_exponent=8.9)))
    / 2e-3
    array([[ 0.        +0.j        ,  0.        +0.j        ,
             0.        +0.j        ,  0.        +0.j        ],
           [ 0.        +0.j        , -1.49391547+0.j        ,
             0.28531247+0.39269892j,  0.        +0.j        ],
           [ 0.        +0.j        , -0.28531247+0.39269892j,
            -1.49391547+0.j        ,  0.        +0.j        ],
           [ 0.        +0.j        ,  0.        +0.j        ,
             0.        +0.j        ,  0.        +0.j        ]])

  */

  EXPECT_EQ(grad.index, 5);
  EXPECT_EQ(grad.params[0], "h2");
  EXPECT_NEAR(grad.grad_gates[0].matrix[0], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[1], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[2], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[3], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[4], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[5], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[6], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[7], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[8], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[9], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[10], -1.49391, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[11], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[12], 0.285312, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[13], 0.392698, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[14], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[15], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[16], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[17], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[18], -0.285312, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[19], 0.392698, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[20], -1.49391, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[21], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[22], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[23], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[24], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[25], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[26], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[27], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[28], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[29], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[30], 0.0, 1e-4);
  EXPECT_NEAR(grad.grad_gates[0].matrix[31], 0.0, 1e-4);
}

TEST(AdjUtilTest, Matrix2Diff) {
  std::array<float, 8> u{1, 2, 3, 4, 5, 6, 7, 8};
  std::array<float, 8> u2{0, 1, 2, 3, 4, 5, 6, 7};
  Matrix2Diff(u, u2);
  for (int i = 0; i < 8; i++) {
    EXPECT_EQ(u2[i], -1);
    EXPECT_EQ(u[i], i + 1);
  }
}

TEST(AdjUtilTest, Matrix4Diff) {
  std::array<float, 32> u;
  std::array<float, 32> u2;

  for (int i = 0; i < 32; i++) {
    u2[i] = i;
    u[i] = i + 1;
  }

  Matrix4Diff(u, u2);
  for (int i = 0; i < 32; i++) {
    EXPECT_EQ(u2[i], -1);
    EXPECT_EQ(u[i], i + 1);
  }
}

}  // namespace
}  // namespace tfq
