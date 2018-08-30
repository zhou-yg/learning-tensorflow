import { CANVAS_WIDTH, CANVAS_HEIGHT } from './game/constants';
import { Runner } from './game';
import * as tf from '@tensorflow/tfjs';

var i = 10;
var ii = 0;

// tf code
// nn - 2
const ni = 50;
const rate = 0.1;

const adamOptimizer = tf.train.adam(rate);

const w1 = tf.variable(tf.zeros([3, 1]));
const b1 = tf.variable(tf.scalar(0));

const w2 = tf.variable(tf.zeros([1, 1]));
const b2 = tf.variable(tf.scalar(0));

//线性公式 y = w * x + b
const f1 = (x) => tf.sigmoid(x.matMul(w1).add(b1));

const f2 = (x) => tf.sigmoid(x.matMul(w2).add(b2));

// loss function
function lossFn(y, yHat) {
  return yHat.sub(y).square().mean();
}

const trainSet = [
  // {
  //   state: [],
  //   result: 0,
  // }
];

var lastState = null;
var trainingSet = [];

const critical = 0.5;

const runner = new Runner('#game', {
  T_REX_COUNT: 1,
  onReset: () => {
    console.log(`reset`);
  },
  onCrash: () => {
    // fix result
    if (lastState.r >= critical) {
      lastState.r = 0;
    } else {
      lastState.r = 1;
    }

    trainSet.push(Object.assign({}, lastState));

    var trainX = (tf.tensor2d(trainSet.map(obj => obj.s.slice())));

    w1.print();

    var trainY = (tf.tensor2d(trainSet.map(obj => [obj.r])));

    for (var i = 0; i < 1; i++) {
      adamOptimizer.minimize(() => {
        const r = lossFn(f2(f1(trainX)), trainY);
        return r;
      });
    }
    setTimeout(() => {
      runner.restart();
    }, 500);
  },
  onRunning: ({tRex, state}) => {
    if (tRex.jumping) {
      return Promise.resolve(1)
    }
    const {obstacleX, obstacleWidth, speed} = state;

    const z2 = tf.tidy(() => {
      const x = tf.tensor2d([[obstacleX, obstacleWidth, speed]]);
      const z1 = f1(x);
      const z2 = f2(z1);
      return z2.dataSync()[0];
    });

    lastState = {
      s: [obstacleX, obstacleWidth, speed],
      r: z2,
    };

    if (z2 >= critical) {
      return Promise.resolve(1)
    } else {
      return Promise.resolve(0)
    }
  },
});
window.runner = runner;
// runner.init();
