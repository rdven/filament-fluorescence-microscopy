��
��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
8
Const
output"dtype"
valuetensor"
dtypetype
�
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

�
Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
.
Identity

input"T
output"T"	
Ttype
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.9.02unknown8��
�
Adam/conv2d_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/conv2d_3/kernel/v
�
*Adam/conv2d_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/kernel/v*&
_output_shapes
: *
dtype0
�
 Adam/conv2d_transpose_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:		  *1
shared_name" Adam/conv2d_transpose_1/kernel/v
�
4Adam/conv2d_transpose_1/kernel/v/Read/ReadVariableOpReadVariableOp Adam/conv2d_transpose_1/kernel/v*&
_output_shapes
:		  *
dtype0
�
Adam/conv2d_transpose/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*/
shared_name Adam/conv2d_transpose/kernel/v
�
2Adam/conv2d_transpose/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose/kernel/v*&
_output_shapes
: @*
dtype0
�
Adam/conv2d_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*'
shared_nameAdam/conv2d_2/kernel/v
�
*Adam/conv2d_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/kernel/v*&
_output_shapes
: @*
dtype0
�
Adam/conv2d_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:		@ *'
shared_nameAdam/conv2d_1/kernel/v
�
*Adam/conv2d_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/kernel/v*&
_output_shapes
:		@ *
dtype0
�
Adam/conv2d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d/kernel/v
�
(Adam/conv2d/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/v*&
_output_shapes
:@*
dtype0
�
Adam/conv2d_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/conv2d_3/kernel/m
�
*Adam/conv2d_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/kernel/m*&
_output_shapes
: *
dtype0
�
 Adam/conv2d_transpose_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:		  *1
shared_name" Adam/conv2d_transpose_1/kernel/m
�
4Adam/conv2d_transpose_1/kernel/m/Read/ReadVariableOpReadVariableOp Adam/conv2d_transpose_1/kernel/m*&
_output_shapes
:		  *
dtype0
�
Adam/conv2d_transpose/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*/
shared_name Adam/conv2d_transpose/kernel/m
�
2Adam/conv2d_transpose/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose/kernel/m*&
_output_shapes
: @*
dtype0
�
Adam/conv2d_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*'
shared_nameAdam/conv2d_2/kernel/m
�
*Adam/conv2d_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/kernel/m*&
_output_shapes
: @*
dtype0
�
Adam/conv2d_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:		@ *'
shared_nameAdam/conv2d_1/kernel/m
�
*Adam/conv2d_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/kernel/m*&
_output_shapes
:		@ *
dtype0
�
Adam/conv2d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d/kernel/m
�
(Adam/conv2d/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/m*&
_output_shapes
:@*
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
�
conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconv2d_3/kernel
{
#conv2d_3/kernel/Read/ReadVariableOpReadVariableOpconv2d_3/kernel*&
_output_shapes
: *
dtype0
�
conv2d_transpose_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:		  **
shared_nameconv2d_transpose_1/kernel
�
-conv2d_transpose_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_1/kernel*&
_output_shapes
:		  *
dtype0
�
conv2d_transpose/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*(
shared_nameconv2d_transpose/kernel
�
+conv2d_transpose/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose/kernel*&
_output_shapes
: @*
dtype0
�
conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @* 
shared_nameconv2d_2/kernel
{
#conv2d_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_2/kernel*&
_output_shapes
: @*
dtype0
�
conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:		@ * 
shared_nameconv2d_1/kernel
{
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
:		@ *
dtype0
~
conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d/kernel
w
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
:@*
dtype0

NoOpNoOp
�@
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�?
value�?B�? B�?
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
encoder
		optimizer


signatures*
.
0
1
2
3
4
5*
.
0
1
2
3
4
5*
* 
�
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
trace_0
trace_1
trace_2
trace_3* 
6
trace_0
trace_1
trace_2
trace_3* 
* 
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
 layer_with_weights-2
 layer-2
!layer_with_weights-3
!layer-3
"layer_with_weights-4
"layer-4
#layer_with_weights-5
#layer-5
$	variables
%trainable_variables
&regularization_losses
'	keras_api
(__call__
*)&call_and_return_all_conditional_losses*
�
*iter

+beta_1

,beta_2
	-decay
.learning_ratem�m�m�m�m�m�v�v�v�v�v�v�*

/serving_default* 
MG
VARIABLE_VALUEconv2d/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEconv2d_1/kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEconv2d_2/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEconv2d_transpose/kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEconv2d_transpose_1/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEconv2d_3/kernel&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
* 

0*

00*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
�
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5__call__
*6&call_and_return_all_conditional_losses

kernel
 7_jit_compiled_convolution_op*
�
8	variables
9trainable_variables
:regularization_losses
;	keras_api
<__call__
*=&call_and_return_all_conditional_losses

kernel
 >_jit_compiled_convolution_op*
�
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
C__call__
*D&call_and_return_all_conditional_losses

kernel
 E_jit_compiled_convolution_op*
�
F	variables
Gtrainable_variables
Hregularization_losses
I	keras_api
J__call__
*K&call_and_return_all_conditional_losses

kernel
 L_jit_compiled_convolution_op*
�
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
Q__call__
*R&call_and_return_all_conditional_losses

kernel
 S_jit_compiled_convolution_op*
�
T	variables
Utrainable_variables
Vregularization_losses
W	keras_api
X__call__
*Y&call_and_return_all_conditional_losses

kernel
 Z_jit_compiled_convolution_op*
.
0
1
2
3
4
5*
.
0
1
2
3
4
5*
* 
�
[non_trainable_variables

\layers
]metrics
^layer_regularization_losses
_layer_metrics
$	variables
%trainable_variables
&regularization_losses
(__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses*
P
`trace_0
atrace_1
btrace_2
ctrace_3
dtrace_4
etrace_5* 
P
ftrace_0
gtrace_1
htrace_2
itrace_3
jtrace_4
ktrace_5* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
8
l	variables
m	keras_api
	ntotal
	ocount*

0*

0*
* 
�
pnon_trainable_variables

qlayers
rmetrics
slayer_regularization_losses
tlayer_metrics
1	variables
2trainable_variables
3regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses*

utrace_0* 

vtrace_0* 
* 

0*

0*
* 
�
wnon_trainable_variables

xlayers
ymetrics
zlayer_regularization_losses
{layer_metrics
8	variables
9trainable_variables
:regularization_losses
<__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses*

|trace_0* 

}trace_0* 
* 

0*

0*
* 
�
~non_trainable_variables

layers
�metrics
 �layer_regularization_losses
�layer_metrics
?	variables
@trainable_variables
Aregularization_losses
C__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 

0*

0*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
F	variables
Gtrainable_variables
Hregularization_losses
J__call__
*K&call_and_return_all_conditional_losses
&K"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 

0*

0*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
M	variables
Ntrainable_variables
Oregularization_losses
Q__call__
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 

0*

0*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
T	variables
Utrainable_variables
Vregularization_losses
X__call__
*Y&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
* 
.
0
1
 2
!3
"4
#5*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

n0
o1*

l	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
pj
VARIABLE_VALUEAdam/conv2d/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/conv2d_1/kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/conv2d_2/kernel/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUEAdam/conv2d_transpose/kernel/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE Adam/conv2d_transpose_1/kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/conv2d_3/kernel/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/conv2d/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/conv2d_1/kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/conv2d_2/kernel/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUEAdam/conv2d_transpose/kernel/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE Adam/conv2d_transpose_1/kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/conv2d_3/kernel/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�
serving_default_input_1Placeholder*+
_output_shapes
:���������PP*
dtype0* 
shape:���������PP
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1conv2d/kernelconv2d_1/kernelconv2d_2/kernelconv2d_transpose/kernelconv2d_transpose_1/kernelconv2d_3/kernel*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������PP*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *+
f&R$
"__inference_signature_wrapper_2093
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!conv2d/kernel/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp#conv2d_2/kernel/Read/ReadVariableOp+conv2d_transpose/kernel/Read/ReadVariableOp-conv2d_transpose_1/kernel/Read/ReadVariableOp#conv2d_3/kernel/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp(Adam/conv2d/kernel/m/Read/ReadVariableOp*Adam/conv2d_1/kernel/m/Read/ReadVariableOp*Adam/conv2d_2/kernel/m/Read/ReadVariableOp2Adam/conv2d_transpose/kernel/m/Read/ReadVariableOp4Adam/conv2d_transpose_1/kernel/m/Read/ReadVariableOp*Adam/conv2d_3/kernel/m/Read/ReadVariableOp(Adam/conv2d/kernel/v/Read/ReadVariableOp*Adam/conv2d_1/kernel/v/Read/ReadVariableOp*Adam/conv2d_2/kernel/v/Read/ReadVariableOp2Adam/conv2d_transpose/kernel/v/Read/ReadVariableOp4Adam/conv2d_transpose_1/kernel/v/Read/ReadVariableOp*Adam/conv2d_3/kernel/v/Read/ReadVariableOpConst*&
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *&
f!R
__inference__traced_save_2761
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d/kernelconv2d_1/kernelconv2d_2/kernelconv2d_transpose/kernelconv2d_transpose_1/kernelconv2d_3/kernel	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/conv2d/kernel/mAdam/conv2d_1/kernel/mAdam/conv2d_2/kernel/mAdam/conv2d_transpose/kernel/m Adam/conv2d_transpose_1/kernel/mAdam/conv2d_3/kernel/mAdam/conv2d/kernel/vAdam/conv2d_1/kernel/vAdam/conv2d_2/kernel/vAdam/conv2d_transpose/kernel/v Adam/conv2d_transpose_1/kernel/vAdam/conv2d_3/kernel/v*%
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *)
f$R"
 __inference__traced_restore_2846��

�	
�
5__inference_filament_reconstructor_layer_call_fn_2110
x!
unknown:@#
	unknown_0:		@ #
	unknown_1: @#
	unknown_2: @#
	unknown_3:		  #
	unknown_4: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������PP*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_filament_reconstructor_layer_call_and_return_conditional_losses_1861w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������PP`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������PP: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
+
_output_shapes
:���������PP

_user_specified_namex
�E
�
D__inference_sequential_layer_call_and_return_conditional_losses_2471

inputs?
%conv2d_conv2d_readvariableop_resource:@A
'conv2d_1_conv2d_readvariableop_resource:		@ A
'conv2d_2_conv2d_readvariableop_resource: @S
9conv2d_transpose_conv2d_transpose_readvariableop_resource: @U
;conv2d_transpose_1_conv2d_transpose_readvariableop_resource:		  A
'conv2d_3_conv2d_readvariableop_resource: 
identity��conv2d/Conv2D/ReadVariableOp�conv2d_1/Conv2D/ReadVariableOp�conv2d_2/Conv2D/ReadVariableOp�conv2d_3/Conv2D/ReadVariableOp�0conv2d_transpose/conv2d_transpose/ReadVariableOp�2conv2d_transpose_1/conv2d_transpose/ReadVariableOpY
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������s

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*/
_output_shapes
:���������PP�
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0�
conv2d/Conv2DConv2DExpandDims:output:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������PP@*
paddingSAME*
strides
e
conv2d/ReluReluconv2d/Conv2D:output:0*
T0*/
_output_shapes
:���������PP@�
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:		@ *
dtype0�
conv2d_1/Conv2DConv2Dconv2d/Relu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������PP *
paddingSAME*
strides
i
conv2d_1/ReluReluconv2d_1/Conv2D:output:0*
T0*/
_output_shapes
:���������PP �
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0�
conv2d_2/Conv2DConv2Dconv2d_1/Relu:activations:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������((@*
paddingSAME*
strides
i
conv2d_2/ReluReluconv2d_2/Conv2D:output:0*
T0*/
_output_shapes
:���������((@a
conv2d_transpose/ShapeShapeconv2d_2/Relu:activations:0*
T0*
_output_shapes
:n
$conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
conv2d_transpose/strided_sliceStridedSliceconv2d_transpose/Shape:output:0-conv2d_transpose/strided_slice/stack:output:0/conv2d_transpose/strided_slice/stack_1:output:0/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :PZ
conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :PZ
conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B : �
conv2d_transpose/stackPack'conv2d_transpose/strided_slice:output:0!conv2d_transpose/stack/1:output:0!conv2d_transpose/stack/2:output:0!conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:p
&conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
 conv2d_transpose/strided_slice_1StridedSliceconv2d_transpose/stack:output:0/conv2d_transpose/strided_slice_1/stack:output:01conv2d_transpose/strided_slice_1/stack_1:output:01conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
0conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOp9conv2d_transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype0�
!conv2d_transpose/conv2d_transposeConv2DBackpropInputconv2d_transpose/stack:output:08conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0conv2d_2/Relu:activations:0*
T0*/
_output_shapes
:���������PP *
paddingSAME*
strides
�
conv2d_transpose/ReluRelu*conv2d_transpose/conv2d_transpose:output:0*
T0*/
_output_shapes
:���������PP k
conv2d_transpose_1/ShapeShape#conv2d_transpose/Relu:activations:0*
T0*
_output_shapes
:p
&conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
 conv2d_transpose_1/strided_sliceStridedSlice!conv2d_transpose_1/Shape:output:0/conv2d_transpose_1/strided_slice/stack:output:01conv2d_transpose_1/strided_slice/stack_1:output:01conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :P\
conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :P\
conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B : �
conv2d_transpose_1/stackPack)conv2d_transpose_1/strided_slice:output:0#conv2d_transpose_1/stack/1:output:0#conv2d_transpose_1/stack/2:output:0#conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
"conv2d_transpose_1/strided_slice_1StridedSlice!conv2d_transpose_1/stack:output:01conv2d_transpose_1/strided_slice_1/stack:output:03conv2d_transpose_1/strided_slice_1/stack_1:output:03conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
2conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
:		  *
dtype0�
#conv2d_transpose_1/conv2d_transposeConv2DBackpropInput!conv2d_transpose_1/stack:output:0:conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0#conv2d_transpose/Relu:activations:0*
T0*/
_output_shapes
:���������PP *
paddingSAME*
strides
�
conv2d_transpose_1/ReluRelu,conv2d_transpose_1/conv2d_transpose:output:0*
T0*/
_output_shapes
:���������PP �
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
conv2d_3/Conv2DConv2D%conv2d_transpose_1/Relu:activations:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������PP*
paddingSAME*
strides
i
conv2d_3/ReluReluconv2d_3/Conv2D:output:0*
T0*/
_output_shapes
:���������PPr
IdentityIdentityconv2d_3/Relu:activations:0^NoOp*
T0*/
_output_shapes
:���������PP�
NoOpNoOp^conv2d/Conv2D/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp1^conv2d_transpose/conv2d_transpose/ReadVariableOp3^conv2d_transpose_1/conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������PP: : : : : : 2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2d
0conv2d_transpose/conv2d_transpose/ReadVariableOp0conv2d_transpose/conv2d_transpose/ReadVariableOp2h
2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2conv2d_transpose_1/conv2d_transpose/ReadVariableOp:S O
+
_output_shapes
:���������PP
 
_user_specified_nameinputs
�
�
B__inference_conv2d_2_layer_call_and_return_conditional_losses_2572

inputs8
conv2d_readvariableop_resource: @
identity��Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������((@*
paddingSAME*
strides
W
ReluReluConv2D:output:0*
T0*/
_output_shapes
:���������((@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������((@^
NoOpNoOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������PP : 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������PP 
 
_user_specified_nameinputs
�E
�
D__inference_sequential_layer_call_and_return_conditional_losses_1846

inputs?
%conv2d_conv2d_readvariableop_resource:@A
'conv2d_1_conv2d_readvariableop_resource:		@ A
'conv2d_2_conv2d_readvariableop_resource: @S
9conv2d_transpose_conv2d_transpose_readvariableop_resource: @U
;conv2d_transpose_1_conv2d_transpose_readvariableop_resource:		  A
'conv2d_3_conv2d_readvariableop_resource: 
identity��conv2d/Conv2D/ReadVariableOp�conv2d_1/Conv2D/ReadVariableOp�conv2d_2/Conv2D/ReadVariableOp�conv2d_3/Conv2D/ReadVariableOp�0conv2d_transpose/conv2d_transpose/ReadVariableOp�2conv2d_transpose_1/conv2d_transpose/ReadVariableOpY
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������s

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*/
_output_shapes
:���������PP�
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0�
conv2d/Conv2DConv2DExpandDims:output:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������PP@*
paddingSAME*
strides
e
conv2d/ReluReluconv2d/Conv2D:output:0*
T0*/
_output_shapes
:���������PP@�
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:		@ *
dtype0�
conv2d_1/Conv2DConv2Dconv2d/Relu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������PP *
paddingSAME*
strides
i
conv2d_1/ReluReluconv2d_1/Conv2D:output:0*
T0*/
_output_shapes
:���������PP �
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0�
conv2d_2/Conv2DConv2Dconv2d_1/Relu:activations:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������((@*
paddingSAME*
strides
i
conv2d_2/ReluReluconv2d_2/Conv2D:output:0*
T0*/
_output_shapes
:���������((@a
conv2d_transpose/ShapeShapeconv2d_2/Relu:activations:0*
T0*
_output_shapes
:n
$conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
conv2d_transpose/strided_sliceStridedSliceconv2d_transpose/Shape:output:0-conv2d_transpose/strided_slice/stack:output:0/conv2d_transpose/strided_slice/stack_1:output:0/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :PZ
conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :PZ
conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B : �
conv2d_transpose/stackPack'conv2d_transpose/strided_slice:output:0!conv2d_transpose/stack/1:output:0!conv2d_transpose/stack/2:output:0!conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:p
&conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
 conv2d_transpose/strided_slice_1StridedSliceconv2d_transpose/stack:output:0/conv2d_transpose/strided_slice_1/stack:output:01conv2d_transpose/strided_slice_1/stack_1:output:01conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
0conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOp9conv2d_transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype0�
!conv2d_transpose/conv2d_transposeConv2DBackpropInputconv2d_transpose/stack:output:08conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0conv2d_2/Relu:activations:0*
T0*/
_output_shapes
:���������PP *
paddingSAME*
strides
�
conv2d_transpose/ReluRelu*conv2d_transpose/conv2d_transpose:output:0*
T0*/
_output_shapes
:���������PP k
conv2d_transpose_1/ShapeShape#conv2d_transpose/Relu:activations:0*
T0*
_output_shapes
:p
&conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
 conv2d_transpose_1/strided_sliceStridedSlice!conv2d_transpose_1/Shape:output:0/conv2d_transpose_1/strided_slice/stack:output:01conv2d_transpose_1/strided_slice/stack_1:output:01conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :P\
conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :P\
conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B : �
conv2d_transpose_1/stackPack)conv2d_transpose_1/strided_slice:output:0#conv2d_transpose_1/stack/1:output:0#conv2d_transpose_1/stack/2:output:0#conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
"conv2d_transpose_1/strided_slice_1StridedSlice!conv2d_transpose_1/stack:output:01conv2d_transpose_1/strided_slice_1/stack:output:03conv2d_transpose_1/strided_slice_1/stack_1:output:03conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
2conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
:		  *
dtype0�
#conv2d_transpose_1/conv2d_transposeConv2DBackpropInput!conv2d_transpose_1/stack:output:0:conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0#conv2d_transpose/Relu:activations:0*
T0*/
_output_shapes
:���������PP *
paddingSAME*
strides
�
conv2d_transpose_1/ReluRelu,conv2d_transpose_1/conv2d_transpose:output:0*
T0*/
_output_shapes
:���������PP �
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
conv2d_3/Conv2DConv2D%conv2d_transpose_1/Relu:activations:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������PP*
paddingSAME*
strides
i
conv2d_3/ReluReluconv2d_3/Conv2D:output:0*
T0*/
_output_shapes
:���������PPr
IdentityIdentityconv2d_3/Relu:activations:0^NoOp*
T0*/
_output_shapes
:���������PP�
NoOpNoOp^conv2d/Conv2D/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp1^conv2d_transpose/conv2d_transpose/ReadVariableOp3^conv2d_transpose_1/conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������PP: : : : : : 2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2d
0conv2d_transpose/conv2d_transpose/ReadVariableOp0conv2d_transpose/conv2d_transpose/ReadVariableOp2h
2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2conv2d_transpose_1/conv2d_transpose/ReadVariableOp:S O
+
_output_shapes
:���������PP
 
_user_specified_nameinputs
�
�
B__inference_conv2d_3_layer_call_and_return_conditional_losses_2663

inputs8
conv2d_readvariableop_resource: 
identity��Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������PP*
paddingSAME*
strides
W
ReluReluConv2D:output:0*
T0*/
_output_shapes
:���������PPi
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������PP^
NoOpNoOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������PP : 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������PP 
 
_user_specified_nameinputs
�
�
'__inference_conv2d_3_layer_call_fn_2655

inputs!
unknown: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������PP*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_conv2d_3_layer_call_and_return_conditional_losses_1615w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������PP`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������PP : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������PP 
 
_user_specified_nameinputs
�
�
D__inference_sequential_layer_call_and_return_conditional_losses_1784
input_1%
conv2d_1765:@'
conv2d_1_1768:		@ '
conv2d_2_1771: @/
conv2d_transpose_1774: @1
conv2d_transpose_1_1777:		  '
conv2d_3_1780: 
identity��conv2d/StatefulPartitionedCall� conv2d_1/StatefulPartitionedCall� conv2d_2/StatefulPartitionedCall� conv2d_3/StatefulPartitionedCall�(conv2d_transpose/StatefulPartitionedCall�*conv2d_transpose_1/StatefulPartitionedCall�
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_1765*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������PP@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_1573�
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0conv2d_1_1768*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������PP *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_1585�
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0conv2d_2_1771*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������((@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_conv2d_2_layer_call_and_return_conditional_losses_1597�
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0conv2d_transpose_1774*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������PP *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_conv2d_transpose_layer_call_and_return_conditional_losses_1513�
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:0conv2d_transpose_1_1777*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������PP *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_1553�
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_1/StatefulPartitionedCall:output:0conv2d_3_1780*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������PP*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_conv2d_3_layer_call_and_return_conditional_losses_1615�
IdentityIdentity)conv2d_3/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������PP�
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall)^conv2d_transpose/StatefulPartitionedCall+^conv2d_transpose_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������PP: : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2T
(conv2d_transpose/StatefulPartitionedCall(conv2d_transpose/StatefulPartitionedCall2X
*conv2d_transpose_1/StatefulPartitionedCall*conv2d_transpose_1/StatefulPartitionedCall:X T
/
_output_shapes
:���������PP
!
_user_specified_name	input_1
�
�
J__inference_conv2d_transpose_layer_call_and_return_conditional_losses_2610

inputsB
(conv2d_transpose_readvariableop_resource: @
identity��conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B : y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype0�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+��������������������������� *
paddingSAME*
strides
s
ReluReluconv2d_transpose:output:0*
T0*A
_output_shapes/
-:+��������������������������� {
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+��������������������������� h
NoOpNoOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:+���������������������������@: 2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�C
�
D__inference_sequential_layer_call_and_return_conditional_losses_2415

inputs?
%conv2d_conv2d_readvariableop_resource:@A
'conv2d_1_conv2d_readvariableop_resource:		@ A
'conv2d_2_conv2d_readvariableop_resource: @S
9conv2d_transpose_conv2d_transpose_readvariableop_resource: @U
;conv2d_transpose_1_conv2d_transpose_readvariableop_resource:		  A
'conv2d_3_conv2d_readvariableop_resource: 
identity��conv2d/Conv2D/ReadVariableOp�conv2d_1/Conv2D/ReadVariableOp�conv2d_2/Conv2D/ReadVariableOp�conv2d_3/Conv2D/ReadVariableOp�0conv2d_transpose/conv2d_transpose/ReadVariableOp�2conv2d_transpose_1/conv2d_transpose/ReadVariableOp�
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0�
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������PP@*
paddingSAME*
strides
e
conv2d/ReluReluconv2d/Conv2D:output:0*
T0*/
_output_shapes
:���������PP@�
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:		@ *
dtype0�
conv2d_1/Conv2DConv2Dconv2d/Relu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������PP *
paddingSAME*
strides
i
conv2d_1/ReluReluconv2d_1/Conv2D:output:0*
T0*/
_output_shapes
:���������PP �
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0�
conv2d_2/Conv2DConv2Dconv2d_1/Relu:activations:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������((@*
paddingSAME*
strides
i
conv2d_2/ReluReluconv2d_2/Conv2D:output:0*
T0*/
_output_shapes
:���������((@a
conv2d_transpose/ShapeShapeconv2d_2/Relu:activations:0*
T0*
_output_shapes
:n
$conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
conv2d_transpose/strided_sliceStridedSliceconv2d_transpose/Shape:output:0-conv2d_transpose/strided_slice/stack:output:0/conv2d_transpose/strided_slice/stack_1:output:0/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :PZ
conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :PZ
conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B : �
conv2d_transpose/stackPack'conv2d_transpose/strided_slice:output:0!conv2d_transpose/stack/1:output:0!conv2d_transpose/stack/2:output:0!conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:p
&conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
 conv2d_transpose/strided_slice_1StridedSliceconv2d_transpose/stack:output:0/conv2d_transpose/strided_slice_1/stack:output:01conv2d_transpose/strided_slice_1/stack_1:output:01conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
0conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOp9conv2d_transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype0�
!conv2d_transpose/conv2d_transposeConv2DBackpropInputconv2d_transpose/stack:output:08conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0conv2d_2/Relu:activations:0*
T0*/
_output_shapes
:���������PP *
paddingSAME*
strides
�
conv2d_transpose/ReluRelu*conv2d_transpose/conv2d_transpose:output:0*
T0*/
_output_shapes
:���������PP k
conv2d_transpose_1/ShapeShape#conv2d_transpose/Relu:activations:0*
T0*
_output_shapes
:p
&conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
 conv2d_transpose_1/strided_sliceStridedSlice!conv2d_transpose_1/Shape:output:0/conv2d_transpose_1/strided_slice/stack:output:01conv2d_transpose_1/strided_slice/stack_1:output:01conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :P\
conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :P\
conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B : �
conv2d_transpose_1/stackPack)conv2d_transpose_1/strided_slice:output:0#conv2d_transpose_1/stack/1:output:0#conv2d_transpose_1/stack/2:output:0#conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
"conv2d_transpose_1/strided_slice_1StridedSlice!conv2d_transpose_1/stack:output:01conv2d_transpose_1/strided_slice_1/stack:output:03conv2d_transpose_1/strided_slice_1/stack_1:output:03conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
2conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
:		  *
dtype0�
#conv2d_transpose_1/conv2d_transposeConv2DBackpropInput!conv2d_transpose_1/stack:output:0:conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0#conv2d_transpose/Relu:activations:0*
T0*/
_output_shapes
:���������PP *
paddingSAME*
strides
�
conv2d_transpose_1/ReluRelu,conv2d_transpose_1/conv2d_transpose:output:0*
T0*/
_output_shapes
:���������PP �
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
conv2d_3/Conv2DConv2D%conv2d_transpose_1/Relu:activations:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������PP*
paddingSAME*
strides
i
conv2d_3/ReluReluconv2d_3/Conv2D:output:0*
T0*/
_output_shapes
:���������PPr
IdentityIdentityconv2d_3/Relu:activations:0^NoOp*
T0*/
_output_shapes
:���������PP�
NoOpNoOp^conv2d/Conv2D/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp1^conv2d_transpose/conv2d_transpose/ReadVariableOp3^conv2d_transpose_1/conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������PP: : : : : : 2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2d
0conv2d_transpose/conv2d_transpose/ReadVariableOp0conv2d_transpose/conv2d_transpose/ReadVariableOp2h
2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2conv2d_transpose_1/conv2d_transpose/ReadVariableOp:W S
/
_output_shapes
:���������PP
 
_user_specified_nameinputs
�:
�

__inference__traced_save_2761
file_prefix,
(savev2_conv2d_kernel_read_readvariableop.
*savev2_conv2d_1_kernel_read_readvariableop.
*savev2_conv2d_2_kernel_read_readvariableop6
2savev2_conv2d_transpose_kernel_read_readvariableop8
4savev2_conv2d_transpose_1_kernel_read_readvariableop.
*savev2_conv2d_3_kernel_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop3
/savev2_adam_conv2d_kernel_m_read_readvariableop5
1savev2_adam_conv2d_1_kernel_m_read_readvariableop5
1savev2_adam_conv2d_2_kernel_m_read_readvariableop=
9savev2_adam_conv2d_transpose_kernel_m_read_readvariableop?
;savev2_adam_conv2d_transpose_1_kernel_m_read_readvariableop5
1savev2_adam_conv2d_3_kernel_m_read_readvariableop3
/savev2_adam_conv2d_kernel_v_read_readvariableop5
1savev2_adam_conv2d_1_kernel_v_read_readvariableop5
1savev2_adam_conv2d_2_kernel_v_read_readvariableop=
9savev2_adam_conv2d_transpose_kernel_v_read_readvariableop?
;savev2_adam_conv2d_transpose_1_kernel_v_read_readvariableop5
1savev2_adam_conv2d_3_kernel_v_read_readvariableop
savev2_const

identity_1��MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B �

SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv2d_kernel_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop*savev2_conv2d_2_kernel_read_readvariableop2savev2_conv2d_transpose_kernel_read_readvariableop4savev2_conv2d_transpose_1_kernel_read_readvariableop*savev2_conv2d_3_kernel_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop/savev2_adam_conv2d_kernel_m_read_readvariableop1savev2_adam_conv2d_1_kernel_m_read_readvariableop1savev2_adam_conv2d_2_kernel_m_read_readvariableop9savev2_adam_conv2d_transpose_kernel_m_read_readvariableop;savev2_adam_conv2d_transpose_1_kernel_m_read_readvariableop1savev2_adam_conv2d_3_kernel_m_read_readvariableop/savev2_adam_conv2d_kernel_v_read_readvariableop1savev2_adam_conv2d_1_kernel_v_read_readvariableop1savev2_adam_conv2d_2_kernel_v_read_readvariableop9savev2_adam_conv2d_transpose_kernel_v_read_readvariableop;savev2_adam_conv2d_transpose_1_kernel_v_read_readvariableop1savev2_adam_conv2d_3_kernel_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *(
dtypes
2	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*�
_input_shapes�
�: :@:		@ : @: @:		  : : : : : : : : :@:		@ : @: @:		  : :@:		@ : @: @:		  : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:@:,(
&
_output_shapes
:		@ :,(
&
_output_shapes
: @:,(
&
_output_shapes
: @:,(
&
_output_shapes
:		  :,(
&
_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
:@:,(
&
_output_shapes
:		@ :,(
&
_output_shapes
: @:,(
&
_output_shapes
: @:,(
&
_output_shapes
:		  :,(
&
_output_shapes
: :,(
&
_output_shapes
:@:,(
&
_output_shapes
:		@ :,(
&
_output_shapes
: @:,(
&
_output_shapes
: @:,(
&
_output_shapes
:		  :,(
&
_output_shapes
: :

_output_shapes
: 
�

�
P__inference_filament_reconstructor_layer_call_and_return_conditional_losses_2051
input_1)
sequential_2037:@)
sequential_2039:		@ )
sequential_2041: @)
sequential_2043: @)
sequential_2045:		  )
sequential_2047: 
identity��"sequential/StatefulPartitionedCall�
"sequential/StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_2037sequential_2039sequential_2041sequential_2043sequential_2045sequential_2047*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������PP*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_1846�
IdentityIdentity+sequential/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������PPk
NoOpNoOp#^sequential/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������PP: : : : : : 2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall:T P
+
_output_shapes
:���������PP
!
_user_specified_name	input_1
�
�
B__inference_conv2d_1_layer_call_and_return_conditional_losses_2557

inputs8
conv2d_readvariableop_resource:		@ 
identity��Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:		@ *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������PP *
paddingSAME*
strides
W
ReluReluConv2D:output:0*
T0*/
_output_shapes
:���������PP i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������PP ^
NoOpNoOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������PP@: 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������PP@
 
_user_specified_nameinputs
�
�
'__inference_conv2d_2_layer_call_fn_2564

inputs!
unknown: @
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������((@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_conv2d_2_layer_call_and_return_conditional_losses_1597w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������((@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������PP : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������PP 
 
_user_specified_nameinputs
�
�
B__inference_conv2d_3_layer_call_and_return_conditional_losses_1615

inputs8
conv2d_readvariableop_resource: 
identity��Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������PP*
paddingSAME*
strides
W
ReluReluConv2D:output:0*
T0*/
_output_shapes
:���������PPi
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������PP^
NoOpNoOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������PP : 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������PP 
 
_user_specified_nameinputs
�	
�
)__inference_sequential_layer_call_fn_2273

inputs!
unknown:@#
	unknown_0:		@ #
	unknown_1: @#
	unknown_2: @#
	unknown_3:		  #
	unknown_4: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������PP*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_1708w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������PP`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������PP: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������PP
 
_user_specified_nameinputs
�	
�
)__inference_sequential_layer_call_fn_2290

inputs!
unknown:@#
	unknown_0:		@ #
	unknown_1: @#
	unknown_2: @#
	unknown_3:		  #
	unknown_4: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������PP*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_1846w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������PP`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������PP: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������PP
 
_user_specified_nameinputs
�
�
%__inference_conv2d_layer_call_fn_2534

inputs!
unknown:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������PP@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_1573w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������PP@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������PP: 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������PP
 
_user_specified_nameinputs
�	
�
"__inference_signature_wrapper_2093
input_1!
unknown:@#
	unknown_0:		@ #
	unknown_1: @#
	unknown_2: @#
	unknown_3:		  #
	unknown_4: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������PP*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *(
f#R!
__inference__wrapped_model_1478w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������PP`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������PP: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:���������PP
!
_user_specified_name	input_1
�
�
J__inference_conv2d_transpose_layer_call_and_return_conditional_losses_1513

inputsB
(conv2d_transpose_readvariableop_resource: @
identity��conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B : y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype0�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+��������������������������� *
paddingSAME*
strides
s
ReluReluconv2d_transpose:output:0*
T0*A
_output_shapes/
-:+��������������������������� {
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+��������������������������� h
NoOpNoOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:+���������������������������@: 2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�	
�
)__inference_sequential_layer_call_fn_2256

inputs!
unknown:@#
	unknown_0:		@ #
	unknown_1: @#
	unknown_2: @#
	unknown_3:		  #
	unknown_4: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������PP*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_1620w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������PP`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������PP: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������PP
 
_user_specified_nameinputs
�h
�
__inference__wrapped_model_1478
input_1a
Gfilament_reconstructor_sequential_conv2d_conv2d_readvariableop_resource:@c
Ifilament_reconstructor_sequential_conv2d_1_conv2d_readvariableop_resource:		@ c
Ifilament_reconstructor_sequential_conv2d_2_conv2d_readvariableop_resource: @u
[filament_reconstructor_sequential_conv2d_transpose_conv2d_transpose_readvariableop_resource: @w
]filament_reconstructor_sequential_conv2d_transpose_1_conv2d_transpose_readvariableop_resource:		  c
Ifilament_reconstructor_sequential_conv2d_3_conv2d_readvariableop_resource: 
identity��>filament_reconstructor/sequential/conv2d/Conv2D/ReadVariableOp�@filament_reconstructor/sequential/conv2d_1/Conv2D/ReadVariableOp�@filament_reconstructor/sequential/conv2d_2/Conv2D/ReadVariableOp�@filament_reconstructor/sequential/conv2d_3/Conv2D/ReadVariableOp�Rfilament_reconstructor/sequential/conv2d_transpose/conv2d_transpose/ReadVariableOp�Tfilament_reconstructor/sequential/conv2d_transpose_1/conv2d_transpose/ReadVariableOp{
0filament_reconstructor/sequential/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
,filament_reconstructor/sequential/ExpandDims
ExpandDimsinput_19filament_reconstructor/sequential/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������PP�
>filament_reconstructor/sequential/conv2d/Conv2D/ReadVariableOpReadVariableOpGfilament_reconstructor_sequential_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0�
/filament_reconstructor/sequential/conv2d/Conv2DConv2D5filament_reconstructor/sequential/ExpandDims:output:0Ffilament_reconstructor/sequential/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������PP@*
paddingSAME*
strides
�
-filament_reconstructor/sequential/conv2d/ReluRelu8filament_reconstructor/sequential/conv2d/Conv2D:output:0*
T0*/
_output_shapes
:���������PP@�
@filament_reconstructor/sequential/conv2d_1/Conv2D/ReadVariableOpReadVariableOpIfilament_reconstructor_sequential_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:		@ *
dtype0�
1filament_reconstructor/sequential/conv2d_1/Conv2DConv2D;filament_reconstructor/sequential/conv2d/Relu:activations:0Hfilament_reconstructor/sequential/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������PP *
paddingSAME*
strides
�
/filament_reconstructor/sequential/conv2d_1/ReluRelu:filament_reconstructor/sequential/conv2d_1/Conv2D:output:0*
T0*/
_output_shapes
:���������PP �
@filament_reconstructor/sequential/conv2d_2/Conv2D/ReadVariableOpReadVariableOpIfilament_reconstructor_sequential_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0�
1filament_reconstructor/sequential/conv2d_2/Conv2DConv2D=filament_reconstructor/sequential/conv2d_1/Relu:activations:0Hfilament_reconstructor/sequential/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������((@*
paddingSAME*
strides
�
/filament_reconstructor/sequential/conv2d_2/ReluRelu:filament_reconstructor/sequential/conv2d_2/Conv2D:output:0*
T0*/
_output_shapes
:���������((@�
8filament_reconstructor/sequential/conv2d_transpose/ShapeShape=filament_reconstructor/sequential/conv2d_2/Relu:activations:0*
T0*
_output_shapes
:�
Ffilament_reconstructor/sequential/conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
Hfilament_reconstructor/sequential/conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Hfilament_reconstructor/sequential/conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
@filament_reconstructor/sequential/conv2d_transpose/strided_sliceStridedSliceAfilament_reconstructor/sequential/conv2d_transpose/Shape:output:0Ofilament_reconstructor/sequential/conv2d_transpose/strided_slice/stack:output:0Qfilament_reconstructor/sequential/conv2d_transpose/strided_slice/stack_1:output:0Qfilament_reconstructor/sequential/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask|
:filament_reconstructor/sequential/conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :P|
:filament_reconstructor/sequential/conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :P|
:filament_reconstructor/sequential/conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B : �
8filament_reconstructor/sequential/conv2d_transpose/stackPackIfilament_reconstructor/sequential/conv2d_transpose/strided_slice:output:0Cfilament_reconstructor/sequential/conv2d_transpose/stack/1:output:0Cfilament_reconstructor/sequential/conv2d_transpose/stack/2:output:0Cfilament_reconstructor/sequential/conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:�
Hfilament_reconstructor/sequential/conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: �
Jfilament_reconstructor/sequential/conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Jfilament_reconstructor/sequential/conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Bfilament_reconstructor/sequential/conv2d_transpose/strided_slice_1StridedSliceAfilament_reconstructor/sequential/conv2d_transpose/stack:output:0Qfilament_reconstructor/sequential/conv2d_transpose/strided_slice_1/stack:output:0Sfilament_reconstructor/sequential/conv2d_transpose/strided_slice_1/stack_1:output:0Sfilament_reconstructor/sequential/conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
Rfilament_reconstructor/sequential/conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOp[filament_reconstructor_sequential_conv2d_transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype0�
Cfilament_reconstructor/sequential/conv2d_transpose/conv2d_transposeConv2DBackpropInputAfilament_reconstructor/sequential/conv2d_transpose/stack:output:0Zfilament_reconstructor/sequential/conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0=filament_reconstructor/sequential/conv2d_2/Relu:activations:0*
T0*/
_output_shapes
:���������PP *
paddingSAME*
strides
�
7filament_reconstructor/sequential/conv2d_transpose/ReluReluLfilament_reconstructor/sequential/conv2d_transpose/conv2d_transpose:output:0*
T0*/
_output_shapes
:���������PP �
:filament_reconstructor/sequential/conv2d_transpose_1/ShapeShapeEfilament_reconstructor/sequential/conv2d_transpose/Relu:activations:0*
T0*
_output_shapes
:�
Hfilament_reconstructor/sequential/conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
Jfilament_reconstructor/sequential/conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Jfilament_reconstructor/sequential/conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Bfilament_reconstructor/sequential/conv2d_transpose_1/strided_sliceStridedSliceCfilament_reconstructor/sequential/conv2d_transpose_1/Shape:output:0Qfilament_reconstructor/sequential/conv2d_transpose_1/strided_slice/stack:output:0Sfilament_reconstructor/sequential/conv2d_transpose_1/strided_slice/stack_1:output:0Sfilament_reconstructor/sequential/conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask~
<filament_reconstructor/sequential/conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :P~
<filament_reconstructor/sequential/conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :P~
<filament_reconstructor/sequential/conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B : �
:filament_reconstructor/sequential/conv2d_transpose_1/stackPackKfilament_reconstructor/sequential/conv2d_transpose_1/strided_slice:output:0Efilament_reconstructor/sequential/conv2d_transpose_1/stack/1:output:0Efilament_reconstructor/sequential/conv2d_transpose_1/stack/2:output:0Efilament_reconstructor/sequential/conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:�
Jfilament_reconstructor/sequential/conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: �
Lfilament_reconstructor/sequential/conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Lfilament_reconstructor/sequential/conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Dfilament_reconstructor/sequential/conv2d_transpose_1/strided_slice_1StridedSliceCfilament_reconstructor/sequential/conv2d_transpose_1/stack:output:0Sfilament_reconstructor/sequential/conv2d_transpose_1/strided_slice_1/stack:output:0Ufilament_reconstructor/sequential/conv2d_transpose_1/strided_slice_1/stack_1:output:0Ufilament_reconstructor/sequential/conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
Tfilament_reconstructor/sequential/conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOp]filament_reconstructor_sequential_conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
:		  *
dtype0�
Efilament_reconstructor/sequential/conv2d_transpose_1/conv2d_transposeConv2DBackpropInputCfilament_reconstructor/sequential/conv2d_transpose_1/stack:output:0\filament_reconstructor/sequential/conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0Efilament_reconstructor/sequential/conv2d_transpose/Relu:activations:0*
T0*/
_output_shapes
:���������PP *
paddingSAME*
strides
�
9filament_reconstructor/sequential/conv2d_transpose_1/ReluReluNfilament_reconstructor/sequential/conv2d_transpose_1/conv2d_transpose:output:0*
T0*/
_output_shapes
:���������PP �
@filament_reconstructor/sequential/conv2d_3/Conv2D/ReadVariableOpReadVariableOpIfilament_reconstructor_sequential_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
1filament_reconstructor/sequential/conv2d_3/Conv2DConv2DGfilament_reconstructor/sequential/conv2d_transpose_1/Relu:activations:0Hfilament_reconstructor/sequential/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������PP*
paddingSAME*
strides
�
/filament_reconstructor/sequential/conv2d_3/ReluRelu:filament_reconstructor/sequential/conv2d_3/Conv2D:output:0*
T0*/
_output_shapes
:���������PP�
IdentityIdentity=filament_reconstructor/sequential/conv2d_3/Relu:activations:0^NoOp*
T0*/
_output_shapes
:���������PP�
NoOpNoOp?^filament_reconstructor/sequential/conv2d/Conv2D/ReadVariableOpA^filament_reconstructor/sequential/conv2d_1/Conv2D/ReadVariableOpA^filament_reconstructor/sequential/conv2d_2/Conv2D/ReadVariableOpA^filament_reconstructor/sequential/conv2d_3/Conv2D/ReadVariableOpS^filament_reconstructor/sequential/conv2d_transpose/conv2d_transpose/ReadVariableOpU^filament_reconstructor/sequential/conv2d_transpose_1/conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������PP: : : : : : 2�
>filament_reconstructor/sequential/conv2d/Conv2D/ReadVariableOp>filament_reconstructor/sequential/conv2d/Conv2D/ReadVariableOp2�
@filament_reconstructor/sequential/conv2d_1/Conv2D/ReadVariableOp@filament_reconstructor/sequential/conv2d_1/Conv2D/ReadVariableOp2�
@filament_reconstructor/sequential/conv2d_2/Conv2D/ReadVariableOp@filament_reconstructor/sequential/conv2d_2/Conv2D/ReadVariableOp2�
@filament_reconstructor/sequential/conv2d_3/Conv2D/ReadVariableOp@filament_reconstructor/sequential/conv2d_3/Conv2D/ReadVariableOp2�
Rfilament_reconstructor/sequential/conv2d_transpose/conv2d_transpose/ReadVariableOpRfilament_reconstructor/sequential/conv2d_transpose/conv2d_transpose/ReadVariableOp2�
Tfilament_reconstructor/sequential/conv2d_transpose_1/conv2d_transpose/ReadVariableOpTfilament_reconstructor/sequential/conv2d_transpose_1/conv2d_transpose/ReadVariableOp:T P
+
_output_shapes
:���������PP
!
_user_specified_name	input_1
�

�
P__inference_filament_reconstructor_layer_call_and_return_conditional_losses_2068
input_1)
sequential_2054:@)
sequential_2056:		@ )
sequential_2058: @)
sequential_2060: @)
sequential_2062:		  )
sequential_2064: 
identity��"sequential/StatefulPartitionedCall�
"sequential/StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_2054sequential_2056sequential_2058sequential_2060sequential_2062sequential_2064*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������PP*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_1951�
IdentityIdentity+sequential/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������PPk
NoOpNoOp#^sequential/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������PP: : : : : : 2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall:T P
+
_output_shapes
:���������PP
!
_user_specified_name	input_1
�

�
P__inference_filament_reconstructor_layer_call_and_return_conditional_losses_2002
x)
sequential_1988:@)
sequential_1990:		@ )
sequential_1992: @)
sequential_1994: @)
sequential_1996:		  )
sequential_1998: 
identity��"sequential/StatefulPartitionedCall�
"sequential/StatefulPartitionedCallStatefulPartitionedCallxsequential_1988sequential_1990sequential_1992sequential_1994sequential_1996sequential_1998*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������PP*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_1951�
IdentityIdentity+sequential/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������PPk
NoOpNoOp#^sequential/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������PP: : : : : : 2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall:N J
+
_output_shapes
:���������PP

_user_specified_namex
�
�
@__inference_conv2d_layer_call_and_return_conditional_losses_2542

inputs8
conv2d_readvariableop_resource:@
identity��Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������PP@*
paddingSAME*
strides
W
ReluReluConv2D:output:0*
T0*/
_output_shapes
:���������PP@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������PP@^
NoOpNoOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������PP: 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������PP
 
_user_specified_nameinputs
�
�
D__inference_sequential_layer_call_and_return_conditional_losses_1708

inputs%
conv2d_1689:@'
conv2d_1_1692:		@ '
conv2d_2_1695: @/
conv2d_transpose_1698: @1
conv2d_transpose_1_1701:		  '
conv2d_3_1704: 
identity��conv2d/StatefulPartitionedCall� conv2d_1/StatefulPartitionedCall� conv2d_2/StatefulPartitionedCall� conv2d_3/StatefulPartitionedCall�(conv2d_transpose/StatefulPartitionedCall�*conv2d_transpose_1/StatefulPartitionedCall�
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_1689*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������PP@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_1573�
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0conv2d_1_1692*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������PP *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_1585�
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0conv2d_2_1695*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������((@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_conv2d_2_layer_call_and_return_conditional_losses_1597�
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0conv2d_transpose_1698*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������PP *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_conv2d_transpose_layer_call_and_return_conditional_losses_1513�
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:0conv2d_transpose_1_1701*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������PP *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_1553�
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_1/StatefulPartitionedCall:output:0conv2d_3_1704*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������PP*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_conv2d_3_layer_call_and_return_conditional_losses_1615�
IdentityIdentity)conv2d_3/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������PP�
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall)^conv2d_transpose/StatefulPartitionedCall+^conv2d_transpose_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������PP: : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2T
(conv2d_transpose/StatefulPartitionedCall(conv2d_transpose/StatefulPartitionedCall2X
*conv2d_transpose_1/StatefulPartitionedCall*conv2d_transpose_1/StatefulPartitionedCall:W S
/
_output_shapes
:���������PP
 
_user_specified_nameinputs
�
�
D__inference_sequential_layer_call_and_return_conditional_losses_1620

inputs%
conv2d_1574:@'
conv2d_1_1586:		@ '
conv2d_2_1598: @/
conv2d_transpose_1601: @1
conv2d_transpose_1_1604:		  '
conv2d_3_1616: 
identity��conv2d/StatefulPartitionedCall� conv2d_1/StatefulPartitionedCall� conv2d_2/StatefulPartitionedCall� conv2d_3/StatefulPartitionedCall�(conv2d_transpose/StatefulPartitionedCall�*conv2d_transpose_1/StatefulPartitionedCall�
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_1574*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������PP@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_1573�
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0conv2d_1_1586*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������PP *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_1585�
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0conv2d_2_1598*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������((@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_conv2d_2_layer_call_and_return_conditional_losses_1597�
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0conv2d_transpose_1601*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������PP *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_conv2d_transpose_layer_call_and_return_conditional_losses_1513�
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:0conv2d_transpose_1_1604*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������PP *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_1553�
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_1/StatefulPartitionedCall:output:0conv2d_3_1616*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������PP*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_conv2d_3_layer_call_and_return_conditional_losses_1615�
IdentityIdentity)conv2d_3/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������PP�
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall)^conv2d_transpose/StatefulPartitionedCall+^conv2d_transpose_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������PP: : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2T
(conv2d_transpose/StatefulPartitionedCall(conv2d_transpose/StatefulPartitionedCall2X
*conv2d_transpose_1/StatefulPartitionedCall*conv2d_transpose_1/StatefulPartitionedCall:W S
/
_output_shapes
:���������PP
 
_user_specified_nameinputs
�E
�
D__inference_sequential_layer_call_and_return_conditional_losses_1951

inputs?
%conv2d_conv2d_readvariableop_resource:@A
'conv2d_1_conv2d_readvariableop_resource:		@ A
'conv2d_2_conv2d_readvariableop_resource: @S
9conv2d_transpose_conv2d_transpose_readvariableop_resource: @U
;conv2d_transpose_1_conv2d_transpose_readvariableop_resource:		  A
'conv2d_3_conv2d_readvariableop_resource: 
identity��conv2d/Conv2D/ReadVariableOp�conv2d_1/Conv2D/ReadVariableOp�conv2d_2/Conv2D/ReadVariableOp�conv2d_3/Conv2D/ReadVariableOp�0conv2d_transpose/conv2d_transpose/ReadVariableOp�2conv2d_transpose_1/conv2d_transpose/ReadVariableOpY
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������s

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*/
_output_shapes
:���������PP�
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0�
conv2d/Conv2DConv2DExpandDims:output:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������PP@*
paddingSAME*
strides
e
conv2d/ReluReluconv2d/Conv2D:output:0*
T0*/
_output_shapes
:���������PP@�
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:		@ *
dtype0�
conv2d_1/Conv2DConv2Dconv2d/Relu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������PP *
paddingSAME*
strides
i
conv2d_1/ReluReluconv2d_1/Conv2D:output:0*
T0*/
_output_shapes
:���������PP �
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0�
conv2d_2/Conv2DConv2Dconv2d_1/Relu:activations:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������((@*
paddingSAME*
strides
i
conv2d_2/ReluReluconv2d_2/Conv2D:output:0*
T0*/
_output_shapes
:���������((@a
conv2d_transpose/ShapeShapeconv2d_2/Relu:activations:0*
T0*
_output_shapes
:n
$conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
conv2d_transpose/strided_sliceStridedSliceconv2d_transpose/Shape:output:0-conv2d_transpose/strided_slice/stack:output:0/conv2d_transpose/strided_slice/stack_1:output:0/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :PZ
conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :PZ
conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B : �
conv2d_transpose/stackPack'conv2d_transpose/strided_slice:output:0!conv2d_transpose/stack/1:output:0!conv2d_transpose/stack/2:output:0!conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:p
&conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
 conv2d_transpose/strided_slice_1StridedSliceconv2d_transpose/stack:output:0/conv2d_transpose/strided_slice_1/stack:output:01conv2d_transpose/strided_slice_1/stack_1:output:01conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
0conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOp9conv2d_transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype0�
!conv2d_transpose/conv2d_transposeConv2DBackpropInputconv2d_transpose/stack:output:08conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0conv2d_2/Relu:activations:0*
T0*/
_output_shapes
:���������PP *
paddingSAME*
strides
�
conv2d_transpose/ReluRelu*conv2d_transpose/conv2d_transpose:output:0*
T0*/
_output_shapes
:���������PP k
conv2d_transpose_1/ShapeShape#conv2d_transpose/Relu:activations:0*
T0*
_output_shapes
:p
&conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
 conv2d_transpose_1/strided_sliceStridedSlice!conv2d_transpose_1/Shape:output:0/conv2d_transpose_1/strided_slice/stack:output:01conv2d_transpose_1/strided_slice/stack_1:output:01conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :P\
conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :P\
conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B : �
conv2d_transpose_1/stackPack)conv2d_transpose_1/strided_slice:output:0#conv2d_transpose_1/stack/1:output:0#conv2d_transpose_1/stack/2:output:0#conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
"conv2d_transpose_1/strided_slice_1StridedSlice!conv2d_transpose_1/stack:output:01conv2d_transpose_1/strided_slice_1/stack:output:03conv2d_transpose_1/strided_slice_1/stack_1:output:03conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
2conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
:		  *
dtype0�
#conv2d_transpose_1/conv2d_transposeConv2DBackpropInput!conv2d_transpose_1/stack:output:0:conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0#conv2d_transpose/Relu:activations:0*
T0*/
_output_shapes
:���������PP *
paddingSAME*
strides
�
conv2d_transpose_1/ReluRelu,conv2d_transpose_1/conv2d_transpose:output:0*
T0*/
_output_shapes
:���������PP �
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
conv2d_3/Conv2DConv2D%conv2d_transpose_1/Relu:activations:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������PP*
paddingSAME*
strides
i
conv2d_3/ReluReluconv2d_3/Conv2D:output:0*
T0*/
_output_shapes
:���������PPr
IdentityIdentityconv2d_3/Relu:activations:0^NoOp*
T0*/
_output_shapes
:���������PP�
NoOpNoOp^conv2d/Conv2D/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp1^conv2d_transpose/conv2d_transpose/ReadVariableOp3^conv2d_transpose_1/conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������PP: : : : : : 2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2d
0conv2d_transpose/conv2d_transpose/ReadVariableOp0conv2d_transpose/conv2d_transpose/ReadVariableOp2h
2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2conv2d_transpose_1/conv2d_transpose/ReadVariableOp:S O
+
_output_shapes
:���������PP
 
_user_specified_nameinputs
�E
�
D__inference_sequential_layer_call_and_return_conditional_losses_2527

inputs?
%conv2d_conv2d_readvariableop_resource:@A
'conv2d_1_conv2d_readvariableop_resource:		@ A
'conv2d_2_conv2d_readvariableop_resource: @S
9conv2d_transpose_conv2d_transpose_readvariableop_resource: @U
;conv2d_transpose_1_conv2d_transpose_readvariableop_resource:		  A
'conv2d_3_conv2d_readvariableop_resource: 
identity��conv2d/Conv2D/ReadVariableOp�conv2d_1/Conv2D/ReadVariableOp�conv2d_2/Conv2D/ReadVariableOp�conv2d_3/Conv2D/ReadVariableOp�0conv2d_transpose/conv2d_transpose/ReadVariableOp�2conv2d_transpose_1/conv2d_transpose/ReadVariableOpY
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������s

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*/
_output_shapes
:���������PP�
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0�
conv2d/Conv2DConv2DExpandDims:output:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������PP@*
paddingSAME*
strides
e
conv2d/ReluReluconv2d/Conv2D:output:0*
T0*/
_output_shapes
:���������PP@�
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:		@ *
dtype0�
conv2d_1/Conv2DConv2Dconv2d/Relu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������PP *
paddingSAME*
strides
i
conv2d_1/ReluReluconv2d_1/Conv2D:output:0*
T0*/
_output_shapes
:���������PP �
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0�
conv2d_2/Conv2DConv2Dconv2d_1/Relu:activations:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������((@*
paddingSAME*
strides
i
conv2d_2/ReluReluconv2d_2/Conv2D:output:0*
T0*/
_output_shapes
:���������((@a
conv2d_transpose/ShapeShapeconv2d_2/Relu:activations:0*
T0*
_output_shapes
:n
$conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
conv2d_transpose/strided_sliceStridedSliceconv2d_transpose/Shape:output:0-conv2d_transpose/strided_slice/stack:output:0/conv2d_transpose/strided_slice/stack_1:output:0/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :PZ
conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :PZ
conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B : �
conv2d_transpose/stackPack'conv2d_transpose/strided_slice:output:0!conv2d_transpose/stack/1:output:0!conv2d_transpose/stack/2:output:0!conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:p
&conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
 conv2d_transpose/strided_slice_1StridedSliceconv2d_transpose/stack:output:0/conv2d_transpose/strided_slice_1/stack:output:01conv2d_transpose/strided_slice_1/stack_1:output:01conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
0conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOp9conv2d_transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype0�
!conv2d_transpose/conv2d_transposeConv2DBackpropInputconv2d_transpose/stack:output:08conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0conv2d_2/Relu:activations:0*
T0*/
_output_shapes
:���������PP *
paddingSAME*
strides
�
conv2d_transpose/ReluRelu*conv2d_transpose/conv2d_transpose:output:0*
T0*/
_output_shapes
:���������PP k
conv2d_transpose_1/ShapeShape#conv2d_transpose/Relu:activations:0*
T0*
_output_shapes
:p
&conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
 conv2d_transpose_1/strided_sliceStridedSlice!conv2d_transpose_1/Shape:output:0/conv2d_transpose_1/strided_slice/stack:output:01conv2d_transpose_1/strided_slice/stack_1:output:01conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :P\
conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :P\
conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B : �
conv2d_transpose_1/stackPack)conv2d_transpose_1/strided_slice:output:0#conv2d_transpose_1/stack/1:output:0#conv2d_transpose_1/stack/2:output:0#conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
"conv2d_transpose_1/strided_slice_1StridedSlice!conv2d_transpose_1/stack:output:01conv2d_transpose_1/strided_slice_1/stack:output:03conv2d_transpose_1/strided_slice_1/stack_1:output:03conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
2conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
:		  *
dtype0�
#conv2d_transpose_1/conv2d_transposeConv2DBackpropInput!conv2d_transpose_1/stack:output:0:conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0#conv2d_transpose/Relu:activations:0*
T0*/
_output_shapes
:���������PP *
paddingSAME*
strides
�
conv2d_transpose_1/ReluRelu,conv2d_transpose_1/conv2d_transpose:output:0*
T0*/
_output_shapes
:���������PP �
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
conv2d_3/Conv2DConv2D%conv2d_transpose_1/Relu:activations:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������PP*
paddingSAME*
strides
i
conv2d_3/ReluReluconv2d_3/Conv2D:output:0*
T0*/
_output_shapes
:���������PPr
IdentityIdentityconv2d_3/Relu:activations:0^NoOp*
T0*/
_output_shapes
:���������PP�
NoOpNoOp^conv2d/Conv2D/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp1^conv2d_transpose/conv2d_transpose/ReadVariableOp3^conv2d_transpose_1/conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������PP: : : : : : 2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2d
0conv2d_transpose/conv2d_transpose/ReadVariableOp0conv2d_transpose/conv2d_transpose/ReadVariableOp2h
2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2conv2d_transpose_1/conv2d_transpose/ReadVariableOp:S O
+
_output_shapes
:���������PP
 
_user_specified_nameinputs
�	
�
5__inference_filament_reconstructor_layer_call_fn_1876
input_1!
unknown:@#
	unknown_0:		@ #
	unknown_1: @#
	unknown_2: @#
	unknown_3:		  #
	unknown_4: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������PP*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_filament_reconstructor_layer_call_and_return_conditional_losses_1861w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������PP`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������PP: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:���������PP
!
_user_specified_name	input_1
�
�
L__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_1553

inputsB
(conv2d_transpose_readvariableop_resource:		  
identity��conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B : y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:		  *
dtype0�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+��������������������������� *
paddingSAME*
strides
s
ReluReluconv2d_transpose:output:0*
T0*A
_output_shapes/
-:+��������������������������� {
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+��������������������������� h
NoOpNoOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:+��������������������������� : 2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
�
D__inference_sequential_layer_call_and_return_conditional_losses_1762
input_1%
conv2d_1743:@'
conv2d_1_1746:		@ '
conv2d_2_1749: @/
conv2d_transpose_1752: @1
conv2d_transpose_1_1755:		  '
conv2d_3_1758: 
identity��conv2d/StatefulPartitionedCall� conv2d_1/StatefulPartitionedCall� conv2d_2/StatefulPartitionedCall� conv2d_3/StatefulPartitionedCall�(conv2d_transpose/StatefulPartitionedCall�*conv2d_transpose_1/StatefulPartitionedCall�
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_1743*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������PP@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_1573�
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0conv2d_1_1746*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������PP *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_1585�
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0conv2d_2_1749*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������((@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_conv2d_2_layer_call_and_return_conditional_losses_1597�
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0conv2d_transpose_1752*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������PP *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_conv2d_transpose_layer_call_and_return_conditional_losses_1513�
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:0conv2d_transpose_1_1755*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������PP *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_1553�
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_1/StatefulPartitionedCall:output:0conv2d_3_1758*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������PP*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_conv2d_3_layer_call_and_return_conditional_losses_1615�
IdentityIdentity)conv2d_3/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������PP�
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall)^conv2d_transpose/StatefulPartitionedCall+^conv2d_transpose_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������PP: : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2T
(conv2d_transpose/StatefulPartitionedCall(conv2d_transpose/StatefulPartitionedCall2X
*conv2d_transpose_1/StatefulPartitionedCall*conv2d_transpose_1/StatefulPartitionedCall:X T
/
_output_shapes
:���������PP
!
_user_specified_name	input_1
�
�
B__inference_conv2d_1_layer_call_and_return_conditional_losses_1585

inputs8
conv2d_readvariableop_resource:		@ 
identity��Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:		@ *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������PP *
paddingSAME*
strides
W
ReluReluConv2D:output:0*
T0*/
_output_shapes
:���������PP i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������PP ^
NoOpNoOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������PP@: 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������PP@
 
_user_specified_nameinputs
�

�
P__inference_filament_reconstructor_layer_call_and_return_conditional_losses_1861
x)
sequential_1847:@)
sequential_1849:		@ )
sequential_1851: @)
sequential_1853: @)
sequential_1855:		  )
sequential_1857: 
identity��"sequential/StatefulPartitionedCall�
"sequential/StatefulPartitionedCallStatefulPartitionedCallxsequential_1847sequential_1849sequential_1851sequential_1853sequential_1855sequential_1857*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������PP*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_1846�
IdentityIdentity+sequential/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������PPk
NoOpNoOp#^sequential/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������PP: : : : : : 2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall:N J
+
_output_shapes
:���������PP

_user_specified_namex
�
�
@__inference_conv2d_layer_call_and_return_conditional_losses_1573

inputs8
conv2d_readvariableop_resource:@
identity��Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������PP@*
paddingSAME*
strides
W
ReluReluConv2D:output:0*
T0*/
_output_shapes
:���������PP@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������PP@^
NoOpNoOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������PP: 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������PP
 
_user_specified_nameinputs
�	
�
)__inference_sequential_layer_call_fn_2307

inputs!
unknown:@#
	unknown_0:		@ #
	unknown_1: @#
	unknown_2: @#
	unknown_3:		  #
	unknown_4: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������PP*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_1951w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������PP`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������PP: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������PP
 
_user_specified_nameinputs
�
�
L__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_2648

inputsB
(conv2d_transpose_readvariableop_resource:		  
identity��conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B : y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:		  *
dtype0�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+��������������������������� *
paddingSAME*
strides
s
ReluReluconv2d_transpose:output:0*
T0*A
_output_shapes/
-:+��������������������������� {
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+��������������������������� h
NoOpNoOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:+��������������������������� : 2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�e
�
 __inference__traced_restore_2846
file_prefix8
assignvariableop_conv2d_kernel:@<
"assignvariableop_1_conv2d_1_kernel:		@ <
"assignvariableop_2_conv2d_2_kernel: @D
*assignvariableop_3_conv2d_transpose_kernel: @F
,assignvariableop_4_conv2d_transpose_1_kernel:		  <
"assignvariableop_5_conv2d_3_kernel: &
assignvariableop_6_adam_iter:	 (
assignvariableop_7_adam_beta_1: (
assignvariableop_8_adam_beta_2: '
assignvariableop_9_adam_decay: 0
&assignvariableop_10_adam_learning_rate: #
assignvariableop_11_total: #
assignvariableop_12_count: B
(assignvariableop_13_adam_conv2d_kernel_m:@D
*assignvariableop_14_adam_conv2d_1_kernel_m:		@ D
*assignvariableop_15_adam_conv2d_2_kernel_m: @L
2assignvariableop_16_adam_conv2d_transpose_kernel_m: @N
4assignvariableop_17_adam_conv2d_transpose_1_kernel_m:		  D
*assignvariableop_18_adam_conv2d_3_kernel_m: B
(assignvariableop_19_adam_conv2d_kernel_v:@D
*assignvariableop_20_adam_conv2d_1_kernel_v:		@ D
*assignvariableop_21_adam_conv2d_2_kernel_v: @L
2assignvariableop_22_adam_conv2d_transpose_kernel_v: @N
4assignvariableop_23_adam_conv2d_transpose_1_kernel_v:		  D
*assignvariableop_24_adam_conv2d_3_kernel_v: 
identity_26��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*|
_output_shapesj
h::::::::::::::::::::::::::*(
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOpassignvariableop_conv2d_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp"assignvariableop_1_conv2d_1_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv2d_2_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp*assignvariableop_3_conv2d_transpose_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp,assignvariableop_4_conv2d_transpose_1_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp"assignvariableop_5_conv2d_3_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_iterIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_beta_1Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_beta_2Identity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_decayIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp&assignvariableop_10_adam_learning_rateIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOpassignvariableop_11_totalIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOpassignvariableop_12_countIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp(assignvariableop_13_adam_conv2d_kernel_mIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp*assignvariableop_14_adam_conv2d_1_kernel_mIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp*assignvariableop_15_adam_conv2d_2_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp2assignvariableop_16_adam_conv2d_transpose_kernel_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp4assignvariableop_17_adam_conv2d_transpose_1_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp*assignvariableop_18_adam_conv2d_3_kernel_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp(assignvariableop_19_adam_conv2d_kernel_vIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp*assignvariableop_20_adam_conv2d_1_kernel_vIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp*assignvariableop_21_adam_conv2d_2_kernel_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp2assignvariableop_22_adam_conv2d_transpose_kernel_vIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp4assignvariableop_23_adam_conv2d_transpose_1_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp*assignvariableop_24_adam_conv2d_3_kernel_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 �
Identity_25Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_26IdentityIdentity_25:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_26Identity_26:output:0*G
_input_shapes6
4: : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
'__inference_conv2d_1_layer_call_fn_2549

inputs!
unknown:		@ 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������PP *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_1585w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������PP `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������PP@: 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������PP@
 
_user_specified_nameinputs
�	
�
)__inference_sequential_layer_call_fn_1740
input_1!
unknown:@#
	unknown_0:		@ #
	unknown_1: @#
	unknown_2: @#
	unknown_3:		  #
	unknown_4: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������PP*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_1708w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������PP`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������PP: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:���������PP
!
_user_specified_name	input_1
�P
�
P__inference_filament_reconstructor_layer_call_and_return_conditional_losses_2183
xJ
0sequential_conv2d_conv2d_readvariableop_resource:@L
2sequential_conv2d_1_conv2d_readvariableop_resource:		@ L
2sequential_conv2d_2_conv2d_readvariableop_resource: @^
Dsequential_conv2d_transpose_conv2d_transpose_readvariableop_resource: @`
Fsequential_conv2d_transpose_1_conv2d_transpose_readvariableop_resource:		  L
2sequential_conv2d_3_conv2d_readvariableop_resource: 
identity��'sequential/conv2d/Conv2D/ReadVariableOp�)sequential/conv2d_1/Conv2D/ReadVariableOp�)sequential/conv2d_2/Conv2D/ReadVariableOp�)sequential/conv2d_3/Conv2D/ReadVariableOp�;sequential/conv2d_transpose/conv2d_transpose/ReadVariableOp�=sequential/conv2d_transpose_1/conv2d_transpose/ReadVariableOpd
sequential/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
sequential/ExpandDims
ExpandDimsx"sequential/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������PP�
'sequential/conv2d/Conv2D/ReadVariableOpReadVariableOp0sequential_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0�
sequential/conv2d/Conv2DConv2Dsequential/ExpandDims:output:0/sequential/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������PP@*
paddingSAME*
strides
{
sequential/conv2d/ReluRelu!sequential/conv2d/Conv2D:output:0*
T0*/
_output_shapes
:���������PP@�
)sequential/conv2d_1/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:		@ *
dtype0�
sequential/conv2d_1/Conv2DConv2D$sequential/conv2d/Relu:activations:01sequential/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������PP *
paddingSAME*
strides

sequential/conv2d_1/ReluRelu#sequential/conv2d_1/Conv2D:output:0*
T0*/
_output_shapes
:���������PP �
)sequential/conv2d_2/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0�
sequential/conv2d_2/Conv2DConv2D&sequential/conv2d_1/Relu:activations:01sequential/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������((@*
paddingSAME*
strides

sequential/conv2d_2/ReluRelu#sequential/conv2d_2/Conv2D:output:0*
T0*/
_output_shapes
:���������((@w
!sequential/conv2d_transpose/ShapeShape&sequential/conv2d_2/Relu:activations:0*
T0*
_output_shapes
:y
/sequential/conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1sequential/conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1sequential/conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
)sequential/conv2d_transpose/strided_sliceStridedSlice*sequential/conv2d_transpose/Shape:output:08sequential/conv2d_transpose/strided_slice/stack:output:0:sequential/conv2d_transpose/strided_slice/stack_1:output:0:sequential/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#sequential/conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :Pe
#sequential/conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :Pe
#sequential/conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B : �
!sequential/conv2d_transpose/stackPack2sequential/conv2d_transpose/strided_slice:output:0,sequential/conv2d_transpose/stack/1:output:0,sequential/conv2d_transpose/stack/2:output:0,sequential/conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:{
1sequential/conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3sequential/conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3sequential/conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
+sequential/conv2d_transpose/strided_slice_1StridedSlice*sequential/conv2d_transpose/stack:output:0:sequential/conv2d_transpose/strided_slice_1/stack:output:0<sequential/conv2d_transpose/strided_slice_1/stack_1:output:0<sequential/conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
;sequential/conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOpDsequential_conv2d_transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype0�
,sequential/conv2d_transpose/conv2d_transposeConv2DBackpropInput*sequential/conv2d_transpose/stack:output:0Csequential/conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0&sequential/conv2d_2/Relu:activations:0*
T0*/
_output_shapes
:���������PP *
paddingSAME*
strides
�
 sequential/conv2d_transpose/ReluRelu5sequential/conv2d_transpose/conv2d_transpose:output:0*
T0*/
_output_shapes
:���������PP �
#sequential/conv2d_transpose_1/ShapeShape.sequential/conv2d_transpose/Relu:activations:0*
T0*
_output_shapes
:{
1sequential/conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3sequential/conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3sequential/conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
+sequential/conv2d_transpose_1/strided_sliceStridedSlice,sequential/conv2d_transpose_1/Shape:output:0:sequential/conv2d_transpose_1/strided_slice/stack:output:0<sequential/conv2d_transpose_1/strided_slice/stack_1:output:0<sequential/conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskg
%sequential/conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :Pg
%sequential/conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :Pg
%sequential/conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B : �
#sequential/conv2d_transpose_1/stackPack4sequential/conv2d_transpose_1/strided_slice:output:0.sequential/conv2d_transpose_1/stack/1:output:0.sequential/conv2d_transpose_1/stack/2:output:0.sequential/conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:}
3sequential/conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
5sequential/conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5sequential/conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
-sequential/conv2d_transpose_1/strided_slice_1StridedSlice,sequential/conv2d_transpose_1/stack:output:0<sequential/conv2d_transpose_1/strided_slice_1/stack:output:0>sequential/conv2d_transpose_1/strided_slice_1/stack_1:output:0>sequential/conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
=sequential/conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOpFsequential_conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
:		  *
dtype0�
.sequential/conv2d_transpose_1/conv2d_transposeConv2DBackpropInput,sequential/conv2d_transpose_1/stack:output:0Esequential/conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0.sequential/conv2d_transpose/Relu:activations:0*
T0*/
_output_shapes
:���������PP *
paddingSAME*
strides
�
"sequential/conv2d_transpose_1/ReluRelu7sequential/conv2d_transpose_1/conv2d_transpose:output:0*
T0*/
_output_shapes
:���������PP �
)sequential/conv2d_3/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
sequential/conv2d_3/Conv2DConv2D0sequential/conv2d_transpose_1/Relu:activations:01sequential/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������PP*
paddingSAME*
strides

sequential/conv2d_3/ReluRelu#sequential/conv2d_3/Conv2D:output:0*
T0*/
_output_shapes
:���������PP}
IdentityIdentity&sequential/conv2d_3/Relu:activations:0^NoOp*
T0*/
_output_shapes
:���������PP�
NoOpNoOp(^sequential/conv2d/Conv2D/ReadVariableOp*^sequential/conv2d_1/Conv2D/ReadVariableOp*^sequential/conv2d_2/Conv2D/ReadVariableOp*^sequential/conv2d_3/Conv2D/ReadVariableOp<^sequential/conv2d_transpose/conv2d_transpose/ReadVariableOp>^sequential/conv2d_transpose_1/conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������PP: : : : : : 2R
'sequential/conv2d/Conv2D/ReadVariableOp'sequential/conv2d/Conv2D/ReadVariableOp2V
)sequential/conv2d_1/Conv2D/ReadVariableOp)sequential/conv2d_1/Conv2D/ReadVariableOp2V
)sequential/conv2d_2/Conv2D/ReadVariableOp)sequential/conv2d_2/Conv2D/ReadVariableOp2V
)sequential/conv2d_3/Conv2D/ReadVariableOp)sequential/conv2d_3/Conv2D/ReadVariableOp2z
;sequential/conv2d_transpose/conv2d_transpose/ReadVariableOp;sequential/conv2d_transpose/conv2d_transpose/ReadVariableOp2~
=sequential/conv2d_transpose_1/conv2d_transpose/ReadVariableOp=sequential/conv2d_transpose_1/conv2d_transpose/ReadVariableOp:N J
+
_output_shapes
:���������PP

_user_specified_namex
�
�
B__inference_conv2d_2_layer_call_and_return_conditional_losses_1597

inputs8
conv2d_readvariableop_resource: @
identity��Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������((@*
paddingSAME*
strides
W
ReluReluConv2D:output:0*
T0*/
_output_shapes
:���������((@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������((@^
NoOpNoOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������PP : 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������PP 
 
_user_specified_nameinputs
�
�
1__inference_conv2d_transpose_1_layer_call_fn_2617

inputs!
unknown:		  
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+��������������������������� *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_1553�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+��������������������������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:+��������������������������� : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�C
�
D__inference_sequential_layer_call_and_return_conditional_losses_2361

inputs?
%conv2d_conv2d_readvariableop_resource:@A
'conv2d_1_conv2d_readvariableop_resource:		@ A
'conv2d_2_conv2d_readvariableop_resource: @S
9conv2d_transpose_conv2d_transpose_readvariableop_resource: @U
;conv2d_transpose_1_conv2d_transpose_readvariableop_resource:		  A
'conv2d_3_conv2d_readvariableop_resource: 
identity��conv2d/Conv2D/ReadVariableOp�conv2d_1/Conv2D/ReadVariableOp�conv2d_2/Conv2D/ReadVariableOp�conv2d_3/Conv2D/ReadVariableOp�0conv2d_transpose/conv2d_transpose/ReadVariableOp�2conv2d_transpose_1/conv2d_transpose/ReadVariableOp�
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0�
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������PP@*
paddingSAME*
strides
e
conv2d/ReluReluconv2d/Conv2D:output:0*
T0*/
_output_shapes
:���������PP@�
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:		@ *
dtype0�
conv2d_1/Conv2DConv2Dconv2d/Relu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������PP *
paddingSAME*
strides
i
conv2d_1/ReluReluconv2d_1/Conv2D:output:0*
T0*/
_output_shapes
:���������PP �
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0�
conv2d_2/Conv2DConv2Dconv2d_1/Relu:activations:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������((@*
paddingSAME*
strides
i
conv2d_2/ReluReluconv2d_2/Conv2D:output:0*
T0*/
_output_shapes
:���������((@a
conv2d_transpose/ShapeShapeconv2d_2/Relu:activations:0*
T0*
_output_shapes
:n
$conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
conv2d_transpose/strided_sliceStridedSliceconv2d_transpose/Shape:output:0-conv2d_transpose/strided_slice/stack:output:0/conv2d_transpose/strided_slice/stack_1:output:0/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :PZ
conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :PZ
conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B : �
conv2d_transpose/stackPack'conv2d_transpose/strided_slice:output:0!conv2d_transpose/stack/1:output:0!conv2d_transpose/stack/2:output:0!conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:p
&conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
 conv2d_transpose/strided_slice_1StridedSliceconv2d_transpose/stack:output:0/conv2d_transpose/strided_slice_1/stack:output:01conv2d_transpose/strided_slice_1/stack_1:output:01conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
0conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOp9conv2d_transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype0�
!conv2d_transpose/conv2d_transposeConv2DBackpropInputconv2d_transpose/stack:output:08conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0conv2d_2/Relu:activations:0*
T0*/
_output_shapes
:���������PP *
paddingSAME*
strides
�
conv2d_transpose/ReluRelu*conv2d_transpose/conv2d_transpose:output:0*
T0*/
_output_shapes
:���������PP k
conv2d_transpose_1/ShapeShape#conv2d_transpose/Relu:activations:0*
T0*
_output_shapes
:p
&conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
 conv2d_transpose_1/strided_sliceStridedSlice!conv2d_transpose_1/Shape:output:0/conv2d_transpose_1/strided_slice/stack:output:01conv2d_transpose_1/strided_slice/stack_1:output:01conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :P\
conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :P\
conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B : �
conv2d_transpose_1/stackPack)conv2d_transpose_1/strided_slice:output:0#conv2d_transpose_1/stack/1:output:0#conv2d_transpose_1/stack/2:output:0#conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
"conv2d_transpose_1/strided_slice_1StridedSlice!conv2d_transpose_1/stack:output:01conv2d_transpose_1/strided_slice_1/stack:output:03conv2d_transpose_1/strided_slice_1/stack_1:output:03conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
2conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
:		  *
dtype0�
#conv2d_transpose_1/conv2d_transposeConv2DBackpropInput!conv2d_transpose_1/stack:output:0:conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0#conv2d_transpose/Relu:activations:0*
T0*/
_output_shapes
:���������PP *
paddingSAME*
strides
�
conv2d_transpose_1/ReluRelu,conv2d_transpose_1/conv2d_transpose:output:0*
T0*/
_output_shapes
:���������PP �
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
conv2d_3/Conv2DConv2D%conv2d_transpose_1/Relu:activations:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������PP*
paddingSAME*
strides
i
conv2d_3/ReluReluconv2d_3/Conv2D:output:0*
T0*/
_output_shapes
:���������PPr
IdentityIdentityconv2d_3/Relu:activations:0^NoOp*
T0*/
_output_shapes
:���������PP�
NoOpNoOp^conv2d/Conv2D/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp1^conv2d_transpose/conv2d_transpose/ReadVariableOp3^conv2d_transpose_1/conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������PP: : : : : : 2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2d
0conv2d_transpose/conv2d_transpose/ReadVariableOp0conv2d_transpose/conv2d_transpose/ReadVariableOp2h
2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2conv2d_transpose_1/conv2d_transpose/ReadVariableOp:W S
/
_output_shapes
:���������PP
 
_user_specified_nameinputs
�	
�
)__inference_sequential_layer_call_fn_1635
input_1!
unknown:@#
	unknown_0:		@ #
	unknown_1: @#
	unknown_2: @#
	unknown_3:		  #
	unknown_4: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������PP*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_1620w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������PP`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������PP: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:���������PP
!
_user_specified_name	input_1
�	
�
5__inference_filament_reconstructor_layer_call_fn_2034
input_1!
unknown:@#
	unknown_0:		@ #
	unknown_1: @#
	unknown_2: @#
	unknown_3:		  #
	unknown_4: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������PP*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_filament_reconstructor_layer_call_and_return_conditional_losses_2002w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������PP`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������PP: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:���������PP
!
_user_specified_name	input_1
�P
�
P__inference_filament_reconstructor_layer_call_and_return_conditional_losses_2239
xJ
0sequential_conv2d_conv2d_readvariableop_resource:@L
2sequential_conv2d_1_conv2d_readvariableop_resource:		@ L
2sequential_conv2d_2_conv2d_readvariableop_resource: @^
Dsequential_conv2d_transpose_conv2d_transpose_readvariableop_resource: @`
Fsequential_conv2d_transpose_1_conv2d_transpose_readvariableop_resource:		  L
2sequential_conv2d_3_conv2d_readvariableop_resource: 
identity��'sequential/conv2d/Conv2D/ReadVariableOp�)sequential/conv2d_1/Conv2D/ReadVariableOp�)sequential/conv2d_2/Conv2D/ReadVariableOp�)sequential/conv2d_3/Conv2D/ReadVariableOp�;sequential/conv2d_transpose/conv2d_transpose/ReadVariableOp�=sequential/conv2d_transpose_1/conv2d_transpose/ReadVariableOpd
sequential/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
sequential/ExpandDims
ExpandDimsx"sequential/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������PP�
'sequential/conv2d/Conv2D/ReadVariableOpReadVariableOp0sequential_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0�
sequential/conv2d/Conv2DConv2Dsequential/ExpandDims:output:0/sequential/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������PP@*
paddingSAME*
strides
{
sequential/conv2d/ReluRelu!sequential/conv2d/Conv2D:output:0*
T0*/
_output_shapes
:���������PP@�
)sequential/conv2d_1/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:		@ *
dtype0�
sequential/conv2d_1/Conv2DConv2D$sequential/conv2d/Relu:activations:01sequential/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������PP *
paddingSAME*
strides

sequential/conv2d_1/ReluRelu#sequential/conv2d_1/Conv2D:output:0*
T0*/
_output_shapes
:���������PP �
)sequential/conv2d_2/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0�
sequential/conv2d_2/Conv2DConv2D&sequential/conv2d_1/Relu:activations:01sequential/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������((@*
paddingSAME*
strides

sequential/conv2d_2/ReluRelu#sequential/conv2d_2/Conv2D:output:0*
T0*/
_output_shapes
:���������((@w
!sequential/conv2d_transpose/ShapeShape&sequential/conv2d_2/Relu:activations:0*
T0*
_output_shapes
:y
/sequential/conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1sequential/conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1sequential/conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
)sequential/conv2d_transpose/strided_sliceStridedSlice*sequential/conv2d_transpose/Shape:output:08sequential/conv2d_transpose/strided_slice/stack:output:0:sequential/conv2d_transpose/strided_slice/stack_1:output:0:sequential/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#sequential/conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :Pe
#sequential/conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :Pe
#sequential/conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B : �
!sequential/conv2d_transpose/stackPack2sequential/conv2d_transpose/strided_slice:output:0,sequential/conv2d_transpose/stack/1:output:0,sequential/conv2d_transpose/stack/2:output:0,sequential/conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:{
1sequential/conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3sequential/conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3sequential/conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
+sequential/conv2d_transpose/strided_slice_1StridedSlice*sequential/conv2d_transpose/stack:output:0:sequential/conv2d_transpose/strided_slice_1/stack:output:0<sequential/conv2d_transpose/strided_slice_1/stack_1:output:0<sequential/conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
;sequential/conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOpDsequential_conv2d_transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype0�
,sequential/conv2d_transpose/conv2d_transposeConv2DBackpropInput*sequential/conv2d_transpose/stack:output:0Csequential/conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0&sequential/conv2d_2/Relu:activations:0*
T0*/
_output_shapes
:���������PP *
paddingSAME*
strides
�
 sequential/conv2d_transpose/ReluRelu5sequential/conv2d_transpose/conv2d_transpose:output:0*
T0*/
_output_shapes
:���������PP �
#sequential/conv2d_transpose_1/ShapeShape.sequential/conv2d_transpose/Relu:activations:0*
T0*
_output_shapes
:{
1sequential/conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3sequential/conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3sequential/conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
+sequential/conv2d_transpose_1/strided_sliceStridedSlice,sequential/conv2d_transpose_1/Shape:output:0:sequential/conv2d_transpose_1/strided_slice/stack:output:0<sequential/conv2d_transpose_1/strided_slice/stack_1:output:0<sequential/conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskg
%sequential/conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :Pg
%sequential/conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :Pg
%sequential/conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B : �
#sequential/conv2d_transpose_1/stackPack4sequential/conv2d_transpose_1/strided_slice:output:0.sequential/conv2d_transpose_1/stack/1:output:0.sequential/conv2d_transpose_1/stack/2:output:0.sequential/conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:}
3sequential/conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
5sequential/conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5sequential/conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
-sequential/conv2d_transpose_1/strided_slice_1StridedSlice,sequential/conv2d_transpose_1/stack:output:0<sequential/conv2d_transpose_1/strided_slice_1/stack:output:0>sequential/conv2d_transpose_1/strided_slice_1/stack_1:output:0>sequential/conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
=sequential/conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOpFsequential_conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
:		  *
dtype0�
.sequential/conv2d_transpose_1/conv2d_transposeConv2DBackpropInput,sequential/conv2d_transpose_1/stack:output:0Esequential/conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0.sequential/conv2d_transpose/Relu:activations:0*
T0*/
_output_shapes
:���������PP *
paddingSAME*
strides
�
"sequential/conv2d_transpose_1/ReluRelu7sequential/conv2d_transpose_1/conv2d_transpose:output:0*
T0*/
_output_shapes
:���������PP �
)sequential/conv2d_3/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
sequential/conv2d_3/Conv2DConv2D0sequential/conv2d_transpose_1/Relu:activations:01sequential/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������PP*
paddingSAME*
strides

sequential/conv2d_3/ReluRelu#sequential/conv2d_3/Conv2D:output:0*
T0*/
_output_shapes
:���������PP}
IdentityIdentity&sequential/conv2d_3/Relu:activations:0^NoOp*
T0*/
_output_shapes
:���������PP�
NoOpNoOp(^sequential/conv2d/Conv2D/ReadVariableOp*^sequential/conv2d_1/Conv2D/ReadVariableOp*^sequential/conv2d_2/Conv2D/ReadVariableOp*^sequential/conv2d_3/Conv2D/ReadVariableOp<^sequential/conv2d_transpose/conv2d_transpose/ReadVariableOp>^sequential/conv2d_transpose_1/conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������PP: : : : : : 2R
'sequential/conv2d/Conv2D/ReadVariableOp'sequential/conv2d/Conv2D/ReadVariableOp2V
)sequential/conv2d_1/Conv2D/ReadVariableOp)sequential/conv2d_1/Conv2D/ReadVariableOp2V
)sequential/conv2d_2/Conv2D/ReadVariableOp)sequential/conv2d_2/Conv2D/ReadVariableOp2V
)sequential/conv2d_3/Conv2D/ReadVariableOp)sequential/conv2d_3/Conv2D/ReadVariableOp2z
;sequential/conv2d_transpose/conv2d_transpose/ReadVariableOp;sequential/conv2d_transpose/conv2d_transpose/ReadVariableOp2~
=sequential/conv2d_transpose_1/conv2d_transpose/ReadVariableOp=sequential/conv2d_transpose_1/conv2d_transpose/ReadVariableOp:N J
+
_output_shapes
:���������PP

_user_specified_namex
�	
�
5__inference_filament_reconstructor_layer_call_fn_2127
x!
unknown:@#
	unknown_0:		@ #
	unknown_1: @#
	unknown_2: @#
	unknown_3:		  #
	unknown_4: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������PP*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_filament_reconstructor_layer_call_and_return_conditional_losses_2002w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������PP`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������PP: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
+
_output_shapes
:���������PP

_user_specified_namex
�
�
/__inference_conv2d_transpose_layer_call_fn_2579

inputs!
unknown: @
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+��������������������������� *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_conv2d_transpose_layer_call_and_return_conditional_losses_1513�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+��������������������������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:+���������������������������@: 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
?
input_14
serving_default_input_1:0���������PPD
output_18
StatefulPartitionedCall:0���������PPtensorflow/serving/predict:��
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
encoder
		optimizer


signatures"
_tf_keras_model
J
0
1
2
3
4
5"
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
�
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
trace_0
trace_1
trace_2
trace_32�
5__inference_filament_reconstructor_layer_call_fn_1876
5__inference_filament_reconstructor_layer_call_fn_2110
5__inference_filament_reconstructor_layer_call_fn_2127
5__inference_filament_reconstructor_layer_call_fn_2034�
���
FullArgSpec$
args�
jself
jx

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 ztrace_0ztrace_1ztrace_2ztrace_3
�
trace_0
trace_1
trace_2
trace_32�
P__inference_filament_reconstructor_layer_call_and_return_conditional_losses_2183
P__inference_filament_reconstructor_layer_call_and_return_conditional_losses_2239
P__inference_filament_reconstructor_layer_call_and_return_conditional_losses_2051
P__inference_filament_reconstructor_layer_call_and_return_conditional_losses_2068�
���
FullArgSpec$
args�
jself
jx

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 ztrace_0ztrace_1ztrace_2ztrace_3
�B�
__inference__wrapped_model_1478input_1"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
 layer_with_weights-2
 layer-2
!layer_with_weights-3
!layer-3
"layer_with_weights-4
"layer-4
#layer_with_weights-5
#layer-5
$	variables
%trainable_variables
&regularization_losses
'	keras_api
(__call__
*)&call_and_return_all_conditional_losses"
_tf_keras_sequential
�
*iter

+beta_1

,beta_2
	-decay
.learning_ratem�m�m�m�m�m�v�v�v�v�v�v�"
	optimizer
,
/serving_default"
signature_map
':%@2conv2d/kernel
):'		@ 2conv2d_1/kernel
):' @2conv2d_2/kernel
1:/ @2conv2d_transpose/kernel
3:1		  2conv2d_transpose_1/kernel
):' 2conv2d_3/kernel
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
'
00"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
5__inference_filament_reconstructor_layer_call_fn_1876input_1"�
���
FullArgSpec$
args�
jself
jx

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
5__inference_filament_reconstructor_layer_call_fn_2110x"�
���
FullArgSpec$
args�
jself
jx

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
5__inference_filament_reconstructor_layer_call_fn_2127x"�
���
FullArgSpec$
args�
jself
jx

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
5__inference_filament_reconstructor_layer_call_fn_2034input_1"�
���
FullArgSpec$
args�
jself
jx

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
P__inference_filament_reconstructor_layer_call_and_return_conditional_losses_2183x"�
���
FullArgSpec$
args�
jself
jx

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
P__inference_filament_reconstructor_layer_call_and_return_conditional_losses_2239x"�
���
FullArgSpec$
args�
jself
jx

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
P__inference_filament_reconstructor_layer_call_and_return_conditional_losses_2051input_1"�
���
FullArgSpec$
args�
jself
jx

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
P__inference_filament_reconstructor_layer_call_and_return_conditional_losses_2068input_1"�
���
FullArgSpec$
args�
jself
jx

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5__call__
*6&call_and_return_all_conditional_losses

kernel
 7_jit_compiled_convolution_op"
_tf_keras_layer
�
8	variables
9trainable_variables
:regularization_losses
;	keras_api
<__call__
*=&call_and_return_all_conditional_losses

kernel
 >_jit_compiled_convolution_op"
_tf_keras_layer
�
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
C__call__
*D&call_and_return_all_conditional_losses

kernel
 E_jit_compiled_convolution_op"
_tf_keras_layer
�
F	variables
Gtrainable_variables
Hregularization_losses
I	keras_api
J__call__
*K&call_and_return_all_conditional_losses

kernel
 L_jit_compiled_convolution_op"
_tf_keras_layer
�
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
Q__call__
*R&call_and_return_all_conditional_losses

kernel
 S_jit_compiled_convolution_op"
_tf_keras_layer
�
T	variables
Utrainable_variables
Vregularization_losses
W	keras_api
X__call__
*Y&call_and_return_all_conditional_losses

kernel
 Z_jit_compiled_convolution_op"
_tf_keras_layer
J
0
1
2
3
4
5"
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
�
[non_trainable_variables

\layers
]metrics
^layer_regularization_losses
_layer_metrics
$	variables
%trainable_variables
&regularization_losses
(__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses"
_generic_user_object
�
`trace_0
atrace_1
btrace_2
ctrace_3
dtrace_4
etrace_52�
)__inference_sequential_layer_call_fn_1635
)__inference_sequential_layer_call_fn_2256
)__inference_sequential_layer_call_fn_2273
)__inference_sequential_layer_call_fn_1740
)__inference_sequential_layer_call_fn_2290
)__inference_sequential_layer_call_fn_2307�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 z`trace_0zatrace_1zbtrace_2zctrace_3zdtrace_4zetrace_5
�
ftrace_0
gtrace_1
htrace_2
itrace_3
jtrace_4
ktrace_52�
D__inference_sequential_layer_call_and_return_conditional_losses_2361
D__inference_sequential_layer_call_and_return_conditional_losses_2415
D__inference_sequential_layer_call_and_return_conditional_losses_1762
D__inference_sequential_layer_call_and_return_conditional_losses_1784
D__inference_sequential_layer_call_and_return_conditional_losses_2471
D__inference_sequential_layer_call_and_return_conditional_losses_2527�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 zftrace_0zgtrace_1zhtrace_2zitrace_3zjtrace_4zktrace_5
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
�B�
"__inference_signature_wrapper_2093input_1"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
N
l	variables
m	keras_api
	ntotal
	ocount"
_tf_keras_metric
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
pnon_trainable_variables

qlayers
rmetrics
slayer_regularization_losses
tlayer_metrics
1	variables
2trainable_variables
3regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses"
_generic_user_object
�
utrace_02�
%__inference_conv2d_layer_call_fn_2534�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zutrace_0
�
vtrace_02�
@__inference_conv2d_layer_call_and_return_conditional_losses_2542�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zvtrace_0
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
wnon_trainable_variables

xlayers
ymetrics
zlayer_regularization_losses
{layer_metrics
8	variables
9trainable_variables
:regularization_losses
<__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses"
_generic_user_object
�
|trace_02�
'__inference_conv2d_1_layer_call_fn_2549�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z|trace_0
�
}trace_02�
B__inference_conv2d_1_layer_call_and_return_conditional_losses_2557�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z}trace_0
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
~non_trainable_variables

layers
�metrics
 �layer_regularization_losses
�layer_metrics
?	variables
@trainable_variables
Aregularization_losses
C__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
'__inference_conv2d_2_layer_call_fn_2564�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
B__inference_conv2d_2_layer_call_and_return_conditional_losses_2572�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
F	variables
Gtrainable_variables
Hregularization_losses
J__call__
*K&call_and_return_all_conditional_losses
&K"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
/__inference_conv2d_transpose_layer_call_fn_2579�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
J__inference_conv2d_transpose_layer_call_and_return_conditional_losses_2610�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
M	variables
Ntrainable_variables
Oregularization_losses
Q__call__
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
1__inference_conv2d_transpose_1_layer_call_fn_2617�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
L__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_2648�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
T	variables
Utrainable_variables
Vregularization_losses
X__call__
*Y&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
'__inference_conv2d_3_layer_call_fn_2655�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
B__inference_conv2d_3_layer_call_and_return_conditional_losses_2663�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
J
0
1
 2
!3
"4
#5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_sequential_layer_call_fn_1635input_1"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
)__inference_sequential_layer_call_fn_2256inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
)__inference_sequential_layer_call_fn_2273inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
)__inference_sequential_layer_call_fn_1740input_1"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
)__inference_sequential_layer_call_fn_2290inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
)__inference_sequential_layer_call_fn_2307inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
D__inference_sequential_layer_call_and_return_conditional_losses_2361inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
D__inference_sequential_layer_call_and_return_conditional_losses_2415inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
D__inference_sequential_layer_call_and_return_conditional_losses_1762input_1"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
D__inference_sequential_layer_call_and_return_conditional_losses_1784input_1"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
D__inference_sequential_layer_call_and_return_conditional_losses_2471inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
D__inference_sequential_layer_call_and_return_conditional_losses_2527inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
.
n0
o1"
trackable_list_wrapper
-
l	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
%__inference_conv2d_layer_call_fn_2534inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
@__inference_conv2d_layer_call_and_return_conditional_losses_2542inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
'__inference_conv2d_1_layer_call_fn_2549inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_conv2d_1_layer_call_and_return_conditional_losses_2557inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
'__inference_conv2d_2_layer_call_fn_2564inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_conv2d_2_layer_call_and_return_conditional_losses_2572inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
/__inference_conv2d_transpose_layer_call_fn_2579inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
J__inference_conv2d_transpose_layer_call_and_return_conditional_losses_2610inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
1__inference_conv2d_transpose_1_layer_call_fn_2617inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
L__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_2648inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
'__inference_conv2d_3_layer_call_fn_2655inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_conv2d_3_layer_call_and_return_conditional_losses_2663inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
,:*@2Adam/conv2d/kernel/m
.:,		@ 2Adam/conv2d_1/kernel/m
.:, @2Adam/conv2d_2/kernel/m
6:4 @2Adam/conv2d_transpose/kernel/m
8:6		  2 Adam/conv2d_transpose_1/kernel/m
.:, 2Adam/conv2d_3/kernel/m
,:*@2Adam/conv2d/kernel/v
.:,		@ 2Adam/conv2d_1/kernel/v
.:, @2Adam/conv2d_2/kernel/v
6:4 @2Adam/conv2d_transpose/kernel/v
8:6		  2 Adam/conv2d_transpose_1/kernel/v
.:, 2Adam/conv2d_3/kernel/v�
__inference__wrapped_model_1478{4�1
*�'
%�"
input_1���������PP
� ";�8
6
output_1*�'
output_1���������PP�
B__inference_conv2d_1_layer_call_and_return_conditional_losses_2557k7�4
-�*
(�%
inputs���������PP@
� "-�*
#� 
0���������PP 
� �
'__inference_conv2d_1_layer_call_fn_2549^7�4
-�*
(�%
inputs���������PP@
� " ����������PP �
B__inference_conv2d_2_layer_call_and_return_conditional_losses_2572k7�4
-�*
(�%
inputs���������PP 
� "-�*
#� 
0���������((@
� �
'__inference_conv2d_2_layer_call_fn_2564^7�4
-�*
(�%
inputs���������PP 
� " ����������((@�
B__inference_conv2d_3_layer_call_and_return_conditional_losses_2663k7�4
-�*
(�%
inputs���������PP 
� "-�*
#� 
0���������PP
� �
'__inference_conv2d_3_layer_call_fn_2655^7�4
-�*
(�%
inputs���������PP 
� " ����������PP�
@__inference_conv2d_layer_call_and_return_conditional_losses_2542k7�4
-�*
(�%
inputs���������PP
� "-�*
#� 
0���������PP@
� �
%__inference_conv2d_layer_call_fn_2534^7�4
-�*
(�%
inputs���������PP
� " ����������PP@�
L__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_2648�I�F
?�<
:�7
inputs+��������������������������� 
� "?�<
5�2
0+��������������������������� 
� �
1__inference_conv2d_transpose_1_layer_call_fn_2617�I�F
?�<
:�7
inputs+��������������������������� 
� "2�/+��������������������������� �
J__inference_conv2d_transpose_layer_call_and_return_conditional_losses_2610�I�F
?�<
:�7
inputs+���������������������������@
� "?�<
5�2
0+��������������������������� 
� �
/__inference_conv2d_transpose_layer_call_fn_2579�I�F
?�<
:�7
inputs+���������������������������@
� "2�/+��������������������������� �
P__inference_filament_reconstructor_layer_call_and_return_conditional_losses_2051q8�5
.�+
%�"
input_1���������PP
p 
� "-�*
#� 
0���������PP
� �
P__inference_filament_reconstructor_layer_call_and_return_conditional_losses_2068q8�5
.�+
%�"
input_1���������PP
p
� "-�*
#� 
0���������PP
� �
P__inference_filament_reconstructor_layer_call_and_return_conditional_losses_2183k2�/
(�%
�
x���������PP
p 
� "-�*
#� 
0���������PP
� �
P__inference_filament_reconstructor_layer_call_and_return_conditional_losses_2239k2�/
(�%
�
x���������PP
p
� "-�*
#� 
0���������PP
� �
5__inference_filament_reconstructor_layer_call_fn_1876d8�5
.�+
%�"
input_1���������PP
p 
� " ����������PP�
5__inference_filament_reconstructor_layer_call_fn_2034d8�5
.�+
%�"
input_1���������PP
p
� " ����������PP�
5__inference_filament_reconstructor_layer_call_fn_2110^2�/
(�%
�
x���������PP
p 
� " ����������PP�
5__inference_filament_reconstructor_layer_call_fn_2127^2�/
(�%
�
x���������PP
p
� " ����������PP�
D__inference_sequential_layer_call_and_return_conditional_losses_1762y@�=
6�3
)�&
input_1���������PP
p 

 
� "-�*
#� 
0���������PP
� �
D__inference_sequential_layer_call_and_return_conditional_losses_1784y@�=
6�3
)�&
input_1���������PP
p

 
� "-�*
#� 
0���������PP
� �
D__inference_sequential_layer_call_and_return_conditional_losses_2361x?�<
5�2
(�%
inputs���������PP
p 

 
� "-�*
#� 
0���������PP
� �
D__inference_sequential_layer_call_and_return_conditional_losses_2415x?�<
5�2
(�%
inputs���������PP
p

 
� "-�*
#� 
0���������PP
� �
D__inference_sequential_layer_call_and_return_conditional_losses_2471t;�8
1�.
$�!
inputs���������PP
p 

 
� "-�*
#� 
0���������PP
� �
D__inference_sequential_layer_call_and_return_conditional_losses_2527t;�8
1�.
$�!
inputs���������PP
p

 
� "-�*
#� 
0���������PP
� �
)__inference_sequential_layer_call_fn_1635l@�=
6�3
)�&
input_1���������PP
p 

 
� " ����������PP�
)__inference_sequential_layer_call_fn_1740l@�=
6�3
)�&
input_1���������PP
p

 
� " ����������PP�
)__inference_sequential_layer_call_fn_2256k?�<
5�2
(�%
inputs���������PP
p 

 
� " ����������PP�
)__inference_sequential_layer_call_fn_2273k?�<
5�2
(�%
inputs���������PP
p

 
� " ����������PP�
)__inference_sequential_layer_call_fn_2290g;�8
1�.
$�!
inputs���������PP
p 

 
� " ����������PP�
)__inference_sequential_layer_call_fn_2307g;�8
1�.
$�!
inputs���������PP
p

 
� " ����������PP�
"__inference_signature_wrapper_2093�?�<
� 
5�2
0
input_1%�"
input_1���������PP";�8
6
output_1*�'
output_1���������PP