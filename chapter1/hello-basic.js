const tf = require('@tensorflow/tfjs')

const s5 = tf.tensor1d([5, 2]);
const s7 = tf.tensor1d([7, 3]);

console.log(s5);

const v = tf.variable(s5);

console.log(v);


s5.sub(s7).square().mean().print()
// console.log(s5.sub(s7).square().mean().print())

tf.sigmoid(s5).print();


const d2 = tf.tensor2d([1, 2, 3 ,4 ], [2, 2]);
const w = tf.scalar(5);
tf.tensor2d([0.5, 0.5], [2, 1]).print()
d2.mul(tf.scalar(2)).matMul(tf.tensor2d([0.5, 0.5], [2, 1])).print()
