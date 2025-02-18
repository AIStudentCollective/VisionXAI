"use client";

import React, { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from "@/components/ui/label";


const supportedArchitectures = [
  'ResNet50',
  'VGG16',
  'InceptionV3',
  'Custom'
  // Add more architectures as needed
];

export default function TensorflowSupportPage() {
  const [classNames, setClassNames] = useState<File | null>(null);
  const [weights, setWeights] = useState<File | null>(null);
  const [image, setImage] = useState<File | null>(null);
  const [selectedArchitecture, setSelectedArchitecture] = useState('');
  const [visualizationUrl, setVisualizationUrl] = useState<string | null>(null);
  const [predictedClass, setPredictedClass] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);


  const handleClassNamesUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files) {
      setClassNames(event.target.files[0]);
    }
  };

  const handleWeightsUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files) {
      setWeights(event.target.files[0]);
    }
  };

  const handleImageUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files) {
      setImage(event.target.files[0]);
    }
  };

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setIsLoading(true);
    setError(null);
    setVisualizationUrl(null); // Clear previous visualization
    setPredictedClass(null);

    const formData = new FormData();
    if (classNames) formData.append('class_names', classNames);
    if (weights) formData.append('weights', weights);
    if (image) formData.append('image', image);
    formData.append('architecture', selectedArchitecture);

    try {
      const response = await fetch('/api/tensorflow/heatmap', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const result = await response.json(); //Parse JSON to get error msg
        throw new Error(result.error || 'Network response was not ok');
      }

      const result = await response.json();
      setVisualizationUrl(`data:image/png;base64,${result.image}`);
      setPredictedClass(result.predicted_class);


    } catch (error: any) {
      console.error('Error:', error);
      setError(error.message || "An unexpected error occurred.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="container mx-auto p-4">
      <h1 className="text-2xl font-bold mb-4">TensorFlow Grad-CAM Visualization</h1>

      {isLoading && <p>Loading...</p>}
      {error && <p className="text-red-500">Error: {error}</p>}

      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <Label htmlFor="classNames" className="block mb-2">Upload Class Names (CSV)</Label>
          <Input
            id="classNames"
            type="file"
            accept=".csv"
            onChange={handleClassNamesUpload}
          />
        </div>
        <div>
          <Label htmlFor="weights" className="block mb-2">Upload Weights Architecture</Label>
          <Input
            id="weights"
            type="file"
            onChange={handleWeightsUpload}
          />
        </div>
        <div>
          <Label htmlFor="image" className="block mb-2">Upload Image</Label>
          <Input
            id="image"
            type="file"
            accept="image/*"
            onChange={handleImageUpload}
          />
        </div>
        <div>
          <Label htmlFor="architecture" className="block mb-2">Select Architecture</Label>
          <select
            id="architecture"
            value={selectedArchitecture}
            onChange={(e) => setSelectedArchitecture(e.target.value)}
            className="w-full p-2 border border-gray-300 rounded"
          >
            <option value="">Select an architecture</option>
            {supportedArchitectures.map((arch) => (
              <option key={arch} value={arch}>{arch}</option>
            ))}
          </select>
        </div>
        <Button type="submit">Submit</Button>
      </form>

      {visualizationUrl && (
        <div className="mt-8">
          <h2 className="text-xl font-medium">Grad-CAM Visualization</h2>
          <img
            src={visualizationUrl}
            alt="Grad-CAM Output"
            className="max-w-full max-h-[500px] border rounded shadow-lg"
          />
          {predictedClass && (
            <div className="mt-4 text-sm text-foreground">
              <p>Predicted Class: {predictedClass}</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
