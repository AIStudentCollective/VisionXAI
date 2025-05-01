"use client";

import React, { useState } from "react";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuTrigger,
  DropdownMenuContent,
  DropdownMenuItem,
} from "@/components/ui/dropdown-menu";

const supportedArchitectures = [
  "ResNet50",
  "VGG16",
  "InceptionV3",
  "Transformer (Google Base)",
  "Custom",
];

export default function TensorflowSupportPage() {
  const [step, setStep] = useState<number>(1);
  const [image, setImage] = useState<File | null>(null);
  const [classNames, setClassNames] = useState<File | null>(null);
  const [weights, setWeights] = useState<File | null>(null);
  const [selectedArch, setSelectedArch] = useState<string>("");
  const [visualizationUrl, setVisualizationUrl] = useState<string | null>(null);
  const [predictedClass, setPredictedClass] = useState<string | null>(null);
  const [explanation, setExplanation] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) setImage(e.target.files[0]);
  };
  const handleClassNamesUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) setClassNames(e.target.files[0]);
  };
  const handleWeightsUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) setWeights(e.target.files[0]);
  };

  const isNextDisabled = () => {
    if (step === 1) return !image;
    if (step === 2) return !selectedArch;
    if (step === 3 && selectedArch !== "Transformer (Google Base)")
      return !classNames || !weights;
    return false;
  };

  const handleSubmit = async () => {
    setIsLoading(true);
    setError(null);
    setVisualizationUrl(null);
    setPredictedClass(null);
    setExplanation(null);

    const formData = new FormData();
    if (image) formData.append("image", image);
    formData.append("architecture", selectedArch);
    if (selectedArch !== "Transformer (Google Base)") {
      if (classNames) formData.append("class_names", classNames);
      if (weights) formData.append("weights", weights);
    }

    try {
      const res = await fetch(
        "http://localhost:5001/api/tensorflow/heatmap",
        { method: "POST", body: formData }
      );
      if (!res.ok) {
        const text = await res.text();
        let msg = `Server error: ${res.status}`;
        try {
          const j = JSON.parse(text);
          msg = j.error || msg;
        } catch {}
        throw new Error(msg);
      }
      const json = await res.json();
      setVisualizationUrl(`data:image/png;base64,${json.image}`);
      setPredictedClass(json.predicted_class);
      setExplanation(json.explanation);
      setStep(4);
    } catch (err: any) {
      setError(err.message || "Unexpected error");
    } finally {
      setIsLoading(false);
    }
  };

  const stepTitles = [
    "Upload image for visualization",
    "Select Architecture",
    "Additional Files",
    "Model Visualization",
  ];

  return (
    <div className="min-h-screen bg-black pt-16 px-6 flex justify-center items-center">
      {/* Outer flex: sidebar + card */}
      <div className="flex w-full max-w-6xl mx-auto items-start gap-8">
        {/* ── Left Sidebar ── */}
        <div className="w-1/3 text-white text-sm space-y-2">
          <h2 className="text-xl font-semibold">Name</h2>
          <span className="inline-block bg-gradient-to-r from-[#9333EA] to-[#6366F1] text-white px-3 py-1 rounded-full text-xs font-medium">
            In Progress
          </span>
          <div className="mt-4 space-y-1">
            <p>
              <span className="text-gray-400">Database ID:</span>{" "}
              <span className="text-white">41603</span>
            </p>
            <p>
              <span className="text-gray-400">Date Created:</span>{" "}
              <span className="text-white">2/26/2025</span>
            </p>
            <p>
              <span className="text-gray-400">Institution:</span>{" "}
              <span className="text-white">UC Davis</span>
            </p>
            <p>
              <span className="text-gray-400">Creator:</span>{" "}
              <span className="text-white">rbhiani</span>
            </p>
          </div>

          {/* ── Comments for Step 4 ── */}
          {step === 4 && (
            <div className="mt-8">
              <h3 className="text-lg font-semibold mb-2">Comments</h3>
              <textarea
                className="w-full h-32 bg-gray-200 rounded-lg p-3 resize-none mb-4"
                placeholder="Add a comment…"
              />
              <Button className="bg-transparent border border-[#9333EA] text-white px-6 py-2 rounded-lg">
                Post Comment
              </Button>
            </div>
          )}
        </div>

        {/* ── Right Purple Card ── */}
        <div className="flex-1 flex justify-center">
          <div
            className="
              flex-1 
              max-w-5xl 
              min-h-[32rem] 
              bg-[#150628] 
              border border-[#9333EA] 
              rounded-2xl 
              shadow-[0_0_24px_#9333EA88] 
              p-10 
              space-y-8
            "
          >
            {/* Breadcrumb */}
            <p className="text-sm text-gray-400">
              Name &gt; {stepTitles[step - 1]}
            </p>

            {/* Title */}
            <h2 className="text-2xl font-semibold text-white">
              {stepTitles[step - 1]}
            </h2>

            {/* Step Content */}
            {step === 1 && (
              <div className="flex flex-col items-center justify-center space-y-4">
                <div className="h-60 w-full border-2 border-dashed border-[#9333EA] rounded-xl flex flex-col items-center justify-center p-4 text-sm text-center">
                  <img
                    src="/images/mage_image-upload.svg"
                    alt="Upload Icon"
                    className="w-16 h-16 mb-3"
                  />
                  <p>Choose a file or drag & drop it here</p>
                  <p className="mt-1 text-xs text-gray-400">
                    JPEG, PNG, PDG formats, up to 50MB
                  </p>
                  <label className="mt-3 cursor-pointer bg-white text-black text-sm font-medium px-6 py-2 rounded-lg shadow-sm hover:bg-gray-100 transition">
                    {image ? image.name : "Browse File"}
                    <input
                      type="file"
                      accept="image/*"
                      className="hidden"
                      onChange={handleImageUpload}
                    />
                  </label>
                </div>
              </div>
            )}

            {step === 2 && (
              <div className="flex flex-col items-center justify-center space-y-6">
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <button className="w-full p-2 border rounded-xl shadow-md text-[#DDD] bg-[#1F0B3A]">
                      {selectedArch || "Select an architecture"}
                    </button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent className="w-full max-h-48 overflow-y-auto bg-[#150628] border border-[#9333EA]">
                    {supportedArchitectures.map((arch) => (
                      <DropdownMenuItem
                        key={arch}
                        onSelect={() => setSelectedArch(arch)}
                        className="px-4 py-2 hover:bg-[#2A124A] cursor-pointer"
                      >
                        {arch}
                      </DropdownMenuItem>
                    ))}
                  </DropdownMenuContent>
                </DropdownMenu>
              </div>
            )}

            {step === 3 && (
              <div className="flex flex-col items-center justify-center space-y-4">
                {selectedArch === "Transformer (Google Base)" ? (
                  <p className="text-center text-gray-200">
                    Vision Transformer doesn’t require extra files.
                  </p>
                ) : (
                  <div className="flex gap-4">
                    <Label className="bg-white text-black px-4 py-2 rounded-lg cursor-pointer">
                      {classNames ? classNames.name : "Class Names CSV"}
                      <Input
                        type="file"
                        accept=".csv"
                        className="hidden"
                        onChange={handleClassNamesUpload}
                      />
                    </Label>
                    <Label className="bg-white text-black px-4 py-2 rounded-lg cursor-pointer">
                      {weights ? weights.name : "Weights File"}
                      <Input
                        type="file"
                        className="hidden"
                        onChange={handleWeightsUpload}
                      />
                    </Label>
                  </div>
                )}
              </div>
            )}

            {step === 4 && (
              <div className="space-y-6 text-gray-200">
                {/* Top Row: Locate + Image */}
                <div className="flex justify-between items-start">
                  <button className="text-purple-300 hover:text-white text-base font-medium">
                    Locate in Database &rarr;
                  </button>
                  {visualizationUrl ? (
                    <img
                      src={visualizationUrl}
                      alt="Heatmap"
                      className="w-48 h-48 rounded-lg shadow-lg"
                    />
                  ) : (
                    <p>Loading visualization…</p>
                  )}
                </div>

                {/* Predicted Class */}
                {predictedClass && (
                  <div>
                    <h3 className="text-lg font-semibold text-white mb-1">
                      Predicted class
                    </h3>
                    <p>{predictedClass}</p>
                  </div>
                )}

                {/* Explanation */}
                {explanation && (
                  <div>
                    <h3 className="text-lg font-semibold text-white mb-1">
                      Explanation
                    </h3>
                    <p className="whitespace-pre-line leading-relaxed">
                      {explanation}
                    </p>
                  </div>
                )}

                {/* Save / Export */}
                <div className="flex gap-4 pt-4">
                  <Button className="bg-gradient-to-r from-purple-600 to-indigo-500 px-6 py-2 rounded-lg text-white">
                    Save
                  </Button>
                  <Button className="bg-transparent border border-white text-white px-6 py-2 rounded-lg">
                    Export
                  </Button>
                </div>
              </div>
            )}

            {/* Navigation */}
            <div className="flex justify-between items-center pt-6">
              {step > 1 ? (
                <Button
                  onClick={() => setStep(step - 1)}
                  className="bg-gray-700 px-6 py-2 rounded-lg text-white"
                >
                  Back
                </Button>
              ) : (
                <div />
              )}

              <Button
                onClick={
                  step < 3
                    ? () => setStep(step + 1)
                    : step === 3
                    ? handleSubmit
                    : () => {
                        setStep(1);
                        setImage(null);
                        setClassNames(null);
                        setWeights(null);
                        setSelectedArch("");
                        setVisualizationUrl(null);
                        setPredictedClass(null);
                        setExplanation(null);
                        setError(null);
                      }
                }
                disabled={isNextDisabled() || isLoading}
                className="bg-gradient-to-r from-purple-600 to-indigo-500 px-6 py-2 rounded-lg text-white"
              >
                {step < 3
                  ? "Next"
                  : step === 3
                  ? isLoading
                    ? "Processing…"
                    : "Done"
                  : "New Analysis"}
              </Button>
            </div>

            {error && (
              <p className="mt-4 text-sm text-red-400">{error}</p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
