import { CANVAS_WIDTH, CANVAS_HEIGHT } from './game/constants';
import { Runner } from './game';
import * as tf from '@tensorflow/tfjs';

var i = 10;
var ii = 0;

// tf code
// nn - 2
const ni = 20;
const rate = 1;

const adamOptimizer = tf.train.adam(rate);

const w1 = tf.variable(tf.tensor2d([0, 0, 0],[3,1]));
const b1 = tf.variable(tf.scalar(0));

const w2 = tf.variable(tf.scalar(0));
const b2 = tf.variable(tf.scalar(0));

//线性公式 y = w * x + b
const f1 = (x) => tf.sigmoid(x.matMul(w1).add(b1));

const f2 = (x) => w2.mul(x).add(b2);

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
      lastState.r = 1;
    } else {
      lastState.r = 0;
    }

    trainSet.push(Object.assign({}, lastState));

    var trainX = (tf.tensor2d(trainSet.map(obj => obj.s.slice())));

    w1.print();

    var trainY = (tf.tensor2d(trainSet.map(obj => obj.r), [trainSet.length, 1]));

    for (var i = 0; i < trainSet.length * 10; i++) {
      adamOptimizer.minimize(() => {
        const r = lossFn(f1(trainX), trainY);
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

    const x = tf.tensor2d([[obstacleX, obstacleWidth, speed]]);
    const z1 = f1(x);
    const z2 = z1.dataSync()[0];

    lastState = {
      s: [obstacleX, obstacleWidth, speed],
      r: z2,
    };


    if (z2 >= critical) {
      return Promise.resolve(0)
    } else {
      return Promise.resolve(1)
    }
  },
});
window.runner = runner;
// runner.init();
