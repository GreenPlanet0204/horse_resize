import React, { useState, useRef } from "react";
import cv from "@techstark/opencv-js";
import { Tensor, InferenceSession } from "onnxruntime-web";
import { detectImage, resize, download } from "./utils/detect";
import { clearBoxes } from "./utils/renderBox";
import "./style/App.css";

const App = () => {
  const [session, setSession] = useState(null);
  const [image, setImage] = useState(null);
  const inputImage = useRef(null);
  const imageRef = useRef(null);
  const canvasRef = useRef(null);
  const canvasRef2 = useRef(null);

  // Configs
  const modelName = "yolov8n.onnx";
  const modelInputShape = [1, 3, 640, 640];
  const resultImageShape = [1, 3, 500, 400];
  const topk = 100;
  const iouThreshold = 0.45;
  const scoreThreshold = 0.2;

  // wait until opencv.js initialized
  cv["onRuntimeInitialized"] = async () => {
    // create session
    const [yolov8, nms] = await Promise.all([
      InferenceSession.create(`${process.env.PUBLIC_URL}/model/${modelName}`),
      InferenceSession.create(
        `${process.env.PUBLIC_URL}/model/nms-yolov8.onnx`
      ),
    ]);

    // warmup main model
    const tensor = new Tensor(
      "float32",
      new Float32Array(modelInputShape.reduce((a, b) => a * b)),
      modelInputShape
    );
    await yolov8.run({ images: tensor });

    setSession({ net: yolov8, nms: nms });
  };

  return (
    <div className="App">
      <div className="content">
        <div className="image">
          <img ref={imageRef} src="#" alt="" />
          <canvas
            id="canvas"
            width={modelInputShape[2]}
            height={modelInputShape[3]}
            ref={canvasRef}
          />
        </div>
        <div className="image2">
          <canvas
            id="canvas"
            width={resultImageShape[2]}
            height={resultImageShape[3]}
            ref={canvasRef2}
          />
        </div>
      </div>

      <input
        type="file"
        ref={inputImage}
        accept="image/*"
        style={{ display: "none" }}
        onChange={(e) => {
          // handle next image to detect
          if (image) {
            URL.revokeObjectURL(image);
            setImage(null);
          }

          const url = URL.createObjectURL(e.target.files[0]); // create image url
          imageRef.current.src = url; // set image source
          clearBoxes(canvasRef.current);
          setImage(url);
        }}
      />
      <div className="btn-container">
        <button
          onClick={() => {
            inputImage.current.click();
          }}
        >
          Open local image
        </button>
        <button
          onClick={() => {
            detectImage(
              imageRef.current,
              canvasRef.current,
              session,
              topk,
              iouThreshold,
              scoreThreshold,
              modelInputShape
            );
          }}
        >
          Detect image
        </button>
        <button
          onClick={() => {
            resize(canvasRef2.current, imageRef.current);
          }}
        >
          Resize Image
        </button>
        <button
          onClick={() => {
            download(canvasRef2.current, imageRef.current);
          }}
        >
          Download Image
        </button>
      </div>
    </div>
  );
};

export default App;
