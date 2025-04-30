"use client";

import React, { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';

const supportedArchitectures = [
  'ResNet50',
  'VGG16',
  'InceptionV3',
  'Transformer (Google Base)',
  'Custom',
  // Add more architectures as needed
];

export default function TensorflowSupportPage() {
  const [classNames, setClassNames] = useState<File | null>(null);
  const [weights, setWeights] = useState<File | null>(null);
  const [image, setImage] = useState<File | null>(null);
  const [selectedArchitecture, setSelectedArchitecture] = useState('');
  const [visualizationUrl, setVisualizationUrl] = useState<string | null>(null);
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

    const formData = new FormData();
    if (classNames) formData.append('class_names', classNames);
    if (weights) formData.append('weights', weights);
    if (image) formData.append('image', image);
    formData.append('architecture', selectedArchitecture);

    try {
        const response = await fetch("http://localhost:5001/process-image", {
          method: "POST",
          body: formData,
          mode: "cors",  // <--- This tells the browser it's a cross-origin request
          headers: {
            "Accept": "application/json", 
          },
        });
      
    
        if (!response.ok) {
          const errorText = await response.text(); // Get the error text
          throw new Error(errorText || 'Network response was not ok');
        }
    
        //  Crucial change: Handle the response as a blob, NOT JSON
        const imageBlob = await response.blob();
        const imageUrl = URL.createObjectURL(imageBlob);
        setVisualizationUrl(imageUrl);
    

    } catch (error) {
      console.error('Error:', error);
      if (error instanceof Error) {  // Corrected type guard
          setError(error.message);
      } else {
          setError("An unknown error occurred.");
      }

    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="container mx-auto p-4">
      <h1 className="text-2xl font-bold mb-4">TensorFlow Support</h1>

      {isLoading && <p>Loading...</p>}
      {error && <p className="text-red-500">Error: {error}</p>}

      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label htmlFor="classNames" className="block mb-2">Upload Class Names (CSV)</label>
          <Input
            id="classNames"
            type="file"
            accept=".csv"
            onChange={handleClassNamesUpload}
          />
        </div>
        <div>
          <label htmlFor="weights" className="block mb-2">Upload Weights Architecture</label>
          <Input
            id="weights"
            type="file"
            onChange={handleWeightsUpload}
          />
        </div>
        <div>
          <label htmlFor="image" className="block mb-2">Upload Image</label>
          <Input
            id="image"
            type="file"
            accept="image/*"
            onChange={handleImageUpload}
          />
        </div>
        <div>
          <label htmlFor="architecture" className="block mb-2">Select Architecture</label>
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
        <div>
          <img src={visualizationUrl} alt="Visualization" />
          <a href={visualizationUrl} download="visualization.png">Download Visualization</a>
        </div>
      )}
    </div>
  );
}