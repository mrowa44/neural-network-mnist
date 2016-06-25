import { Layer, Network, Trainer } from 'synaptic';
import mnist from 'mnist';

const TRAINING_SIZE = 1000;
const TEST_SIZE     = 20;

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
  iterations: 50,
  error:      0.01,
  shuffle:    true,
  log:        1,
  cost:       Trainer.cost.CROSS_ENTROPY
});

function getRandomInt(min, max) {
  return Math.floor(Math.random() * (max - min + 1)) + min;
}

const testElement = testSet[getRandomInt(0, TEST_SIZE)];
const { input, output } = testElement;
console.log(myNetwork.activate(input));
console.log(output);
