"use client";

import React, { useState } from "react";
import { SubmitButton } from "@/components/submit-button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

export default function Torch() {
  const [modelName, setModelName] = useState("");
  const [weightsFile, setWeightsFile] = useState<File | null>(null);
  const [targetLayer, setTargetLayer] = useState("");
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [classLabelsFile, setClassLabelsFile] = useState<File | null>(null);
  const [visualizationUrl, setVisualizationUrl] = useState<string | null>(null);
  const [predictionInfo, setPredictionInfo] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleFileUpload =
    (setter: React.Dispatch<React.SetStateAction<File | null>>) =>
    (event: React.ChangeEvent<HTMLInputElement>) => {
      if (event.target.files) {
        setter(event.target.files[0]);
      }
    };

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setIsLoading(true);
    setError(null);

    if (!modelName || !targetLayer || !imageFile || !classLabelsFile) {
      setError("Error: Model name, target layer, image file, and class labels CSV are required.");
      setIsLoading(false);
      return;
    }

    const formData = new FormData();
    formData.append("model_name", modelName);
    formData.append("target_layer", targetLayer);
    formData.append("image_path", imageFile as File);
    formData.append("class_labels_csv", classLabelsFile as File);
    if (weightsFile) formData.append("weights_path", weightsFile);

    try {
      const response = await fetch("/api/pytorch/heatmap", { method: "POST", body: formData });

      if (!response.ok) {
        throw new Error("Failed to generate Grad-CAM visualization.");
      }

      const result = await response.json();
      setVisualizationUrl(`data:image/png;base64,${result.image}`);
      setPredictionInfo(`Predicted Class: ${result.predicted_class} (${result.predicted_probability})`);
    } catch (error) {
      setError("Error processing request. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="container mx-auto p-4">
      <h1 className="text-2xl font-bold mb-4">Grad-CAM Visualization</h1>

      {isLoading && <p>Loading...</p>}
      {error && <p className="text-red-500">Error: {error}</p>}

      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <Label htmlFor="model_name">Model Name</Label>
          <Input
            id="model_name"
            type="text"
            value={modelName}
            onChange={(e) => setModelName(e.target.value)}
            placeholder="e.g., densenet121, resnet50"
            required
          />
        </div>
        <div>
          <Label htmlFor="weights_path">Weights File (Optional)</Label>
          <Input
            id="weights_path"
            type="file"
            accept=".pth,.tar"
            onChange={handleFileUpload(setWeightsFile)}
          />
          <p className="text-sm text-gray-500">Leave blank to use PyTorch pretrained weights</p>
        </div>
        <div>
          <Label htmlFor="target_layer">Target Layer <span className="text-red-500">*</span></Label>
          <Input
            id="target_layer"
            type="text"
            value={targetLayer}
            onChange={(e) => setTargetLayer(e.target.value)}
            placeholder="e.g., features"
            required
          />
        </div>
        <div>
          <Label htmlFor="image_path">Image File <span className="text-red-500">*</span></Label>
          <Input
            id="image_path"
            type="file"
            accept="image/*"
            onChange={handleFileUpload(setImageFile)}
            required
          />
        </div>
        <div>
          <Label htmlFor="class_labels_csv">Class Labels CSV <span className="text-red-500">*</span></Label>
          <Input
            id="class_labels_csv"
            type="file"
            accept=".csv"
            onChange={handleFileUpload(setClassLabelsFile)}
            required
          />
        </div>
        <SubmitButton pendingText="Processing...">Generate</SubmitButton>
      </form>

      {visualizationUrl && (
        <div className="mt-8">
          <h2 className="text-xl font-medium">Heatmap</h2>
          <img
            src={visualizationUrl}
            alt="Grad-CAM Output"
            className="max-w-full max-h-[500px] border rounded shadow-lg"
          />
          <div className="mt-4 text-sm text-foreground">
            {predictionInfo?.split("\n").map((line, idx) => (
              <p key={idx}>{line}</p>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
