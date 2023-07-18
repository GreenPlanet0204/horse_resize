import cv from "@techstark/opencv-js";
import { Tensor } from "onnxruntime-web";
import { renderBoxes } from "./renderBox";
/**
 * Detect Image
 * @param {HTMLImageElement} image Image to detect
 * @param {HTMLCanvasElement} canvas canvas to draw boxes
 * @param {ort.InferenceSession} session YOLOv8 onnxruntime session
 * @param {Number} topk Integer representing the maximum number of boxes to be selected per class
 * @param {Number} iouThreshold Float representing the threshold for deciding whether boxes overlap too much with respect to IOU
 * @param {Number} scoreThreshold Float representing the threshold for deciding when to remove boxes based on score
 * @param {Number[]} inputShape model input shape. Normally in YOLO model [batch, channels, width, height]
 */
let boxes = [];
let image_url = "";
export const detectImage = async (
  image,
  canvas,
  session,
  topk,
  iouThreshold,
  scoreThreshold,
  inputShape
) => {
  const [modelWidth, modelHeight] = inputShape.slice(2);
  const [input, xRatio, yRatio] = preprocessing(image, modelWidth, modelHeight);

  const tensor = new Tensor("float32", input.data32F, inputShape); // to ort.Tensor
  const config = new Tensor(
    "float32",
    new Float32Array([topk, iouThreshold, scoreThreshold])
  ); // nms config tensor
  const { output0 } = await session.net.run({ images: tensor }); // run session and get output layer
  const { selected } = await session.nms.run({
    detection: output0,
    config: config,
  }); // perform nms and filter boxes

  boxes = [];

  // looping through output
  for (let idx = 0; idx < selected.dims[1]; idx++) {
    const data = selected.data.slice(
      idx * selected.dims[2],
      (idx + 1) * selected.dims[2]
    ); // get rows
    const box = data.slice(0, 4);
    const scores = data.slice(4); // classes probability scores
    const score = Math.max(...scores); // maximum probability scores
    const label = scores.indexOf(score); // class id of maximum probability scores

    const [x, y, w, h] = [
      (box[0] - 0.5 * box[2]) * xRatio, // upscale left
      (box[1] - 0.5 * box[3]) * yRatio, // upscale top
      box[2] * xRatio, // upscale width
      box[3] * yRatio, // upscale height
    ]; // keep boxes in maxSize range
    if (label === 17) {
      boxes.push({
        label: label,
        probability: score,
        bounding: [x, y, w, h], // upscale box
      }); // update boxes to draw later
    }
  }
  renderBoxes(canvas, boxes); // Draw boxes
  // input.delete(); // delete unused Mat
};

/**
 * Preprocessing image
 * @param {HTMLImageElement} source image source
 * @param {Number} modelWidth model input width
 * @param {Number} modelHeight model input height
 * @return preprocessed image and configs
 */

const preprocessing = (source, modelWidth, modelHeight) => {
  const mat = cv.imread(source); // read from img tag
  const matC3 = new cv.Mat(mat.rows, mat.cols, cv.CV_8UC3); // new image matrix
  cv.cvtColor(mat, matC3, cv.COLOR_RGBA2BGR); // RGBA to BGR

  // padding image to [n x n] dim
  const maxSize = Math.max(matC3.rows, matC3.cols); // get max size from width and height
  const xPad = maxSize - matC3.cols, // set xPadding
    xRatio = maxSize / matC3.cols; // set xRatio
  const yPad = maxSize - matC3.rows, // set yPadding
    yRatio = maxSize / matC3.rows; // set yRatio
  const matPad = new cv.Mat(); // new mat for padded image
  cv.copyMakeBorder(matC3, matPad, 0, yPad, 0, xPad, cv.BORDER_CONSTANT); // padding black

  const input = cv.blobFromImage(
    matPad,
    1 / 255.0, // normalize
    new cv.Size(modelWidth, modelHeight), // resize to model input size
    new cv.Scalar(0, 0, 0),
    true, // swapRB
    false // crop
  ); // preprocessing image matrix

  // release mat opencv
  mat.delete();
  matC3.delete();
  matPad.delete();

  return [input, xRatio, yRatio];
};

export const resize = (canvas, image) => {
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  let [x, y, w, h] = boxes[0]?.bounding;
  const src = cv.imread(image); // read from img tag
  const [width, height] = [src.cols, src.rows];
  let dst = new cv.Mat();
  [x, y, w, h] = [
    parseInt((x / 640) * width),
    parseInt((y / 640) * height),
    parseInt((w / 640) * width),
    parseInt((h / 640) * height),
  ];
  let rect = new cv.Rect(x, y, w, h);
  dst = src.roi(rect);
  let dsize = new cv.Size(500, 400);
  let res = new cv.Mat();
  // You can try more different parameters
  cv.resize(dst, res, dsize, 0, 0, cv.INTER_AREA);
  cv.imshow(canvas, res);
  // let imgDataUrl = cv.imencode('.jpg', res).data().toString();
  image_url = canvas.toDataURL();

  // Create a new anchor element
  // let anchor = document.createElement('a');
  // anchor.href = imgDataUrl;
  // anchor.download = 'image.jpg'; // Set the desired filename for the image

  // // Append the anchor element to the document
  // document.body.appendChild(anchor);

  // // Trigger a click event on the anchor element to initiate download
  // anchor.click();

  // // Remove the anchor element from the document
  // document.body.removeChild(anchor);
  // result_image = res;
};

export const download = () => {
  let anchor = document.createElement("a");
  anchor.href = image_url;
  anchor.download = "image.png"; // Set the desired filename for the image

  // Append the anchor element to the document
  document.body.appendChild(anchor);

  // Trigger a click event on the anchor element to initiate download
  anchor.click();

  // Remove the anchor element from the document
  document.body.removeChild(anchor);
};
