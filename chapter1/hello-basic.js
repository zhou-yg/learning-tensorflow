const tf = require('@tensorflow/tfjs')

const s5 = tf.scalar(5);

console.log(s5);

s5.print();
console.log('====');
const v1 = tf.tensor1d([0, 1, 2]);
console.log(v1);

v1.print();


tf.ones([3, 5]).print();

const z = tf.zeros([2, 3]);
z.print();

const zvalue = tf.variable(z);

zvalue.print();

const d2 = tf.tensor2d([[1,2,3], [4,5,6]]);

zvalue.assign(d2)

d2.square();

zvalue.print();



const e = tf.tensor2d([[1.0, 2.0], [3.0, 4.0]]);
const f = tf.tensor2d([[5.0, 6.0], [7.0, 8.0]]);

const e_plus_f = e.add(f);
e_plus_f.print();


const x = tf.tensor1d([0, -1, 2, -3]);

x.sigmoid().print();  // or tf.sigmoid(x)

console.log(`sigmoid:`, tf.sigmoid(x).then);

const r =tf.tidy(() => {
  return tf.sigmoid(x);
});

console.log(`r:`);

console.log('====')

const buffer = tf.buffer([2, 2], 'bool');
buffer.set(3, 0, 0);
buffer.set(5, 1, 0);

// Convert the buffer back to a tensor.
buffer.toTensor().print();
console.log(buffer);

//
console.log('====');
tf.truncatedNormal([2, 2]).print();
