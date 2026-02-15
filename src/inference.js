const tfjs = require('@tensorflow/tfjs-node');
const path = require('path');

function loadModel() {
    const modelPath = path.join(__dirname, '../models/model.json'); 
    console.log("Model path:", modelPath);
    return tfjs.loadGraphModel(`file://${modelPath}`)
}

module.exports = {
    loadModel,
    predict
}

function predict(model, imageBuffer) {
    const tensor = tfjs.node
    .decodeJpeg(imageBuffer)
    .resizeNearestNeighbor([150, 150])
    .expandDims()
    .toFloat();

    return model.predict(tensor).data();
}