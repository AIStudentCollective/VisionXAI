"use client";

import React, { useState } from "react";
import { SubmitButton } from "@/components/submit-button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import { DropdownMenu } from "@/components/ui/dropdown-menu";
import { DropdownMenuTrigger } from "@/components/ui/dropdown-menu";
import { DropdownMenuItem } from "@/components/ui/dropdown-menu";
import { DropdownMenuContent } from "@/components/ui/dropdown-menu";

// Added ViT support as in your first version
const supportedArchitectures = [
  'ResNet50',
  'VGG16',
  'InceptionV3',
  'Custom'
];

export default function TensorflowSupportPage() {
  const [step, setStep] = useState(1);
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

  const handleSubmit = async () => {
    setIsLoading(true);
    setError(null);
    setVisualizationUrl(null);
    setPredictedClass(null);

    const formData = new FormData();
    if (image) formData.append('image', image);
    formData.append('architecture', selectedArchitecture);

    // Only add class names & weights if not using ViT
    if (selectedArchitecture !== 'ViT') {
      if (classNames) formData.append('class_names', classNames);
      if (weights) formData.append('weights', weights);
    }

    try {
      const response = await fetch('/api/tensorflow/heatmap', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const result = await response.json();
        throw new Error(result.error || 'Network response was not ok');
      }

      const result = await response.json();
      setVisualizationUrl(`data:image/png;base64,${result.image}`);
      setPredictedClass(result.predicted_class);
      setStep(4); // Advance to results page on success

    } catch (error: any) {
      console.error('Error:', error);
      setError(error.message || "An unexpected error occurred.");
    } finally {
      setIsLoading(false);
    }
  };

  const isNextButtonDisabled = () => {
    if (step === 1) return !image;
    if (step === 2) return !selectedArchitecture;
    if (step === 3) {
      if (selectedArchitecture === 'ViT') return false; // ViT doesn't need additional files
      return !classNames || !weights;
    }
    return false;
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-black pt-16">
      <div className="bg-white p-6 shadow-lg w-[40rem] h-[36rem] relative flex flex-col justify-center items-center rounded-xl">
        {step === 1 && (
          <>
            <h1 className="absolute top-8 left-14 text-4xl font-normal text-black">Upload Image</h1>

            <div className="w-[25rem] h-[22rem] border-2 border-dashed rounded-lg flex flex-col items-center justify-center text-center p-6 mt-24 bg-white border-purple-500">
              <img
                src="/images/mage_image-upload.svg"
                alt="Upload Icon"
                className="w-24 h-24"
              />

              {image ? (
                <p className="bg-[linear-gradient(90deg,#9333EA_0%,#6366F1_100%)] bg-clip-text text-transparent text-lg font-normal mt-2">{image.name}</p>
              ) : (
                <>
                  <p className="bg-[linear-gradient(90deg,#9333EA_0%,#6366F1_100%)] bg-clip-text text-transparent text-lg font-normal mt-2">
                    Drag and drop an image
                  </p>
                  <p className="bg-[linear-gradient(90deg,#9333EA_0%,#6366F1_100%)] bg-clip-text text-transparent text-lg font-normal">or</p>
                </>
              )}

              <label className="cursor-pointer bg-[linear-gradient(90deg,#9333EA_0%,#6366F1_100%)] text-white text-base font-medium py-2 px-14 rounded-xl mt-2 inline-block">
                {image ? "Change file" : "Select file"}
                <input
                  id="image"
                  type="file"
                  accept="image/*"
                  className="hidden"
                  onChange={handleImageUpload}
                  required
                />
              </label>
            </div>

            <Button
              className="mt-auto ml-auto bg-[linear-gradient(90deg,#9333EA_0%,#6366F1_100%)] text-white text-xl font-normal px-10 py-6 rounded-lg"
              onClick={() => setStep(2)}
              disabled={!image}
            >
              Next
            </Button>
          </>
        )}

        {step === 2 && (
          <>
            <h1 className="absolute top-8 left-14 text-4xl font-normal text-black">Select Architecture</h1>

            <div className="flex justify-center items-center w-full mt-28">
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <button className="w-[25rem] p-2 border rounded-xl shadow-md text-[#737373] text-base font-medium bg-white">
                    {selectedArchitecture ? selectedArchitecture : "Select an architecture"}
                  </button>
                </DropdownMenuTrigger>

                <DropdownMenuContent className="w-[25rem] max-h-[200px] overflow-y-auto">
                  {supportedArchitectures.map((architecture) => (
                    <DropdownMenuItem
                      key={architecture}
                      onSelect={() => setSelectedArchitecture(architecture)}
                      className="cursor-pointer px-4 py-2 text-[#737373] text-base font-medium transition-all 
                      hover:!bg-[#F6F2FF] hover:!text-black hover:!border hover:!border-[#4918B2]
                      focus:!bg-[#F6F2FF] focus:!text-black focus:!border focus:!border-[#4918B2] 
                      rounded-md"
                    >
                      {architecture}
                    </DropdownMenuItem>
                  ))}
                </DropdownMenuContent>
              </DropdownMenu>
            </div>

            <div className="flex justify-between items-center mt-auto w-full">
              <Button
                className="mt-6 bg-[linear-gradient(90deg,#9333EA_0%,#6366F1_100%)] text-white text-xl font-normal px-10 py-6 rounded-lg"
                onClick={() => setStep(1)}
              >
                Back
              </Button>
              <Button
                className="mt-6 bg-[linear-gradient(90deg,#9333EA_0%,#6366F1_100%)] text-white text-xl font-normal px-10 py-6 rounded-lg"
                onClick={() => setStep(3)}
                disabled={!selectedArchitecture}
              >
                Next
              </Button>
            </div>
          </>
        )}

        {step === 3 && (
          <>
            <h1 className="absolute top-8 left-14 text-4xl font-normal text-black">
              {selectedArchitecture === 'ViT' ? 'Ready to Process' : 'Additional Files'}
            </h1>

            {selectedArchitecture === 'ViT' ? (
              <div className="flex flex-col items-center justify-center mt-28 text-center">
                <div className="text-lg text-gray-700 mb-8">
                  <p>Vision Transformer (ViT) models don't require additional files.</p>
                  <p className="mt-2">Click "Done" to process your image with ViT attention rollout.</p>
                </div>
              </div>
            ) : (
              <div className="flex justify-center items-start w-full mt-28 gap-6">
                {/* Class Names CSV Button */}
                <div className="relative inline-block bg-[linear-gradient(90deg,#9333EA_0%,#6366F1_100%)] p-[1px] rounded-[13px]">
                  <Label className="flex items-center justify-center cursor-pointer bg-white text-black text-base font-medium px-6 py-2 rounded-xl">
                    {classNames ? classNames.name : "Class Names CSV"}
                    <Input
                      id="classNames"
                      type="file"
                      accept=".csv"
                      className="hidden"
                      onChange={handleClassNamesUpload}
                    />
                  </Label>
                </div>

                {/* Weights File Button */}
                <div className="relative inline-block bg-[linear-gradient(90deg,#9333EA_0%,#6366F1_100%)] p-[1px] rounded-[13px]">
                  <Label className="flex items-center justify-center cursor-pointer bg-white text-black text-base font-medium px-6 py-2 rounded-xl">
                    {weights ? weights.name : "Weights File"}
                    <Input
                      id="weights"
                      type="file"
                      className="hidden"
                      onChange={handleWeightsUpload}
                    />
                  </Label>
                </div>
              </div>
            )}

            <div className="flex justify-between items-center mt-auto w-full">
              <Button
                className="bg-[linear-gradient(90deg,#9333EA_0%,#6366F1_100%)] text-white text-xl font-normal px-10 py-6 rounded-lg"
                onClick={() => setStep(2)}
              >
                Back
              </Button>

              <Button
                className="bg-[linear-gradient(90deg,#9333EA_0%,#6366F1_100%)] text-white text-xl font-normal px-10 py-6 rounded-lg"
                onClick={handleSubmit}
                disabled={isNextButtonDisabled() || isLoading}
              >
                {isLoading ? "Processing..." : "Done"}
              </Button>
            </div>
          </>
        )}

        {step === 4 && (
          <>
            <h2 className="absolute top-8 left-14 text-4xl font-normal text-black">Generated Visualization</h2>
            {visualizationUrl ? (
              <>
                <div className="flex flex-col justify-center items-center gap-4 mt-28">
                  <img
                    src={visualizationUrl}
                    alt={selectedArchitecture === "ViT" ? "Attention Rollout" : "Grad-CAM Output"}
                    className="max-w-[35vh] max-h-[35vh] border rounded shadow-lg"
                  />

                  <div className="text-sm text-foreground text-center">
                    {predictedClass && <p>Predicted Class: {predictedClass}</p>}
                  </div>
                </div>
              </>
            ) : (
              <p className="text-gray-500 mt-4">Processing visualization...</p>
            )}

            <div className="mt-auto ml-auto">
              <Button
                className="bg-[linear-gradient(90deg,#9333EA_0%,#6366F1_100%)] text-white text-xl font-normal px-10 py-6 rounded-lg"
                onClick={() => {
                  // Reset state to start over
                  setStep(1);
                  setClassNames(null);
                  setWeights(null);
                  setImage(null);
                  setSelectedArchitecture('');
                  setVisualizationUrl(null);
                  setPredictedClass(null);
                }}
              >
                New Analysis
              </Button>
            </div>
          </>
        )}

        {isLoading && (
          <div className="absolute inset-0 flex items-center justify-center bg-white bg-opacity-80">
            <div className="bg-white p-6 rounded-lg shadow-md">
              <p className="text-xl font-medium">Processing...</p>
            </div>
          </div>
        )}
        
        {error && (
          <div className="absolute bottom-4 left-1/2 -translate-x-1/2 bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded shadow-md">
            <p>{error}</p>
            <button 
              className="absolute top-1 right-1 text-red-500"
              onClick={() => setError(null)}
            >
              ×
            </button>
          </div>
        )}
      </div>
    </div>
  );
}