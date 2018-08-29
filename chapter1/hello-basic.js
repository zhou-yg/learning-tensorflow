const tf = require('@tensorflow/tfjs')

const s5 = tf.tensor1d([5, 2]);
const s7 = tf.tensor1d([7, 3]);

console.log(s5);

const v = tf.variable(s5);

console.log(v);


s5.sub(s7).square().mean().print()
// console.log(s5.sub(s7).square().mean().print())

tf.sigmoid(s5).print();


const d1 = tf.tensor2d([[0, 1, 2]]);
const d2 = tf.tensor2d([[tf.scalar(0),1,2],[2,3,4]]);

d2.print();

const w = tf.ones([3, 1]);

w.print();
d1.matMul(w).print();
d2.matMul(w).add(tf.scalar(2)).print();

tf.randomNormal([2, 6]).print();
