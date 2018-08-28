const tf = require('@tensorflow/tfjs')

function predict(input) {
  return tf.tidy(() => {

    const x = tf.scalar(input);

    return x.mul(tf.scalar(2));
  });
}


const r = predict(3);

console.log(r);

r.print();
