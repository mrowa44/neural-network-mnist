import { Layer, Network, Trainer } from 'synaptic';
import mnist from 'mnist';

const TRAINING_SIZE = 1000;
const TEST_SIZE     = 20;
const getRandomInt = (min, max) => Math.floor(Math.random() * (max - min + 1)) + min;

// Generate training and test sets
const mnistSet    = mnist.set(TRAINING_SIZE, TEST_SIZE);
const trainingSet = mnistSet.training;
const testSet     = mnistSet.test;

// Size of each image is 28x28px, the number of pixels the network has to take
// as input is 28 x 28 = 784
const inputLayer = new Layer(784);
// Hidden layer of 100 neurons
const hiddenLayer = new Layer(100);
// The digits should be assigned to one of ten classes (0-9)
const outputLayer = new Layer(10);

inputLayer.project(hiddenLayer);
hiddenLayer.project(outputLayer);

const myNetwork = new Network({
  input:  inputLayer,
  hidden: [hiddenLayer],
  output: outputLayer
});

const trainer = new Trainer(myNetwork);
trainer.train(trainingSet, {
  rate:       0.2,
  iterations: 20,
  error:      0.01,
  shuffle:    true,
  log:        1,
  cost:       Trainer.cost.CROSS_ENTROPY
});

const testElement = testSet[getRandomInt(0, TEST_SIZE)];
const { input, output } = testElement;
const guess = myNetwork.activate(input);

const maximum = guess.reduce((prev, curr) => Math.max(prev, curr));
const nominators = guess.map(x => Math.exp(x - maximum));
const denominator = nominators.reduce((prev, curr) => prev + curr);
const softmax = nominators.map(e => e / denominator);

let maxIndex = 0;
softmax.reduce((prev, curr, i) => {
  if (curr > prev) {
    maxIndex = i;
    return curr;
  } else {
    return prev;
  }
});

const result = [];
for (let i = 0; i < guess.length; i++) {
  i === maxIndex ? result.push(1) : result.push(0);
}

const equalsOne    = elem => elem === 1;
const guessedDigit = result.findIndex(equalsOne);
const testDigit    = output.findIndex(equalsOne);

if (guessedDigit === testDigit) {
  console.log(`Yay! ${guessedDigit} was guessed correctly!`);
} else {
  console.log(`Oh no, I thought it was ${guessedDigit}, but really it was ${testDigit} :(`);
}
process.exit();
