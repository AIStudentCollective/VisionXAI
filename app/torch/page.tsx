"use client";

import React, { useState } from "react";
import { SubmitButton } from "@/components/submit-button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import * as DropdownMenu from "@radix-ui/react-dropdown-menu"
import { Check, ChevronDown, ChevronUp } from "lucide-react";
import { cn } from "@/lib/utils";

const architectures = [
  "resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
  "vgg11", "vgg13", "vgg16", "vgg19",
  "alexnet", "inception_v3",
  "densenet121", "densenet169", "densenet201", "densenet161",
  "mobilenet_v2", "shufflenet_v2_x0_5", "shufflenet_v2_x1_0",
];


export default function Torch() {
  const [step, setStep] = useState(1);
  const [modelName, setModelName] = useState("");
  const [weightsFile, setWeightsFile] = useState<File | null>(null);
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [classLabelsFile, setClassLabelsFile] = useState<File | null>(null);
  const [visualizationUrl, setVisualizationUrl] = useState<string | null>(null);
  const [predictionInfo, setPredictionInfo] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [architectureSelected, setArchitectureSelected] = useState(false);
  const [customModelFile, setCustomModelFile] = useState<File | null>(null);
  const [weightsSelected, setWeightsSelected] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const [search, setSearch] = React.useState(""); // Search input state
  const [open, setOpen] = useState(false);

  const handleFileUpload = (setter: React.Dispatch<React.SetStateAction<File | null>>, setFlag?: React.Dispatch<React.SetStateAction<boolean>>) => 
    (event: React.ChangeEvent<HTMLInputElement>) => {
      if (event.target.files) {
        setter(event.target.files[0]);
        if (setFlag) setFlag(true);
      }
    };

  const handleSubmit = async () => {
    setIsLoading(true);
    setError(null);

    if (!modelName || !imageFile || !classLabelsFile) {
      setError("Error: Model name, image file, and class labels CSV are required.");
      setIsLoading(false);
      return;
    }

    const formData = new FormData();
    formData.append("model_name", modelName);
    formData.append("image_path", imageFile as File);
    formData.append("class_labels_csv", classLabelsFile as File);
    if (!weightsSelected && weightsFile) {
      formData.append("weights_path", weightsFile);
    }

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

  // Drag-and-drop handlers
  const handleDragOver = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setDragActive(true);
  };

  const handleDragLeave = () => {
    setDragActive(false);
  };

  const handleDrop = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setDragActive(false);
    if (event.dataTransfer.files && event.dataTransfer.files.length > 0) {
      setImageFile(event.dataTransfer.files[0]);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-black pt-[4rem]">
      <div className="bg-white p-6 shadow-lg w-[40rem] h-[36rem] relative flex flex-col justify-center items-center">
      {step === 1 && (
        <>
            <h1 className="absolute top-8 left-14 text-[36px] font-normal text-black">Upload Image</h1>

            <div
              className={`w-[25rem] h-[22rem] border-2 border-dashed rounded-lg flex flex-col items-center justify-center text-center p-6 mt-24 transition-all duration-200 ${
                dragActive ? "bg-[#F6F2FF] border-purple-600" : "bg-white border-purple-500"
              }`}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
            >

              <img
                src="/images/mage_image-upload.svg"
                alt="Upload Icon"
                className="w-24 h-24"
              />

              {imageFile ? (
                <p className="bg-[linear-gradient(90deg,#9333EA_0%,#6366F1_100%)] bg-clip-text text-transparent text-[18px] font-normal mt-2">{imageFile.name}</p>
              ) : (
                <>
                  <p className="bg-[linear-gradient(90deg,#9333EA_0%,#6366F1_100%)] bg-clip-text text-transparent text-[18px] font-normal mt-2">
                    Drag and drop an image
                  </p>
                  <p className="bg-[linear-gradient(90deg,#9333EA_0%,#6366F1_100%)] bg-clip-text text-transparent text-[18px] font-normal">or</p>
                </>
              )}

              <label className="cursor-pointer bg-[linear-gradient(90deg,#9333EA_0%,#6366F1_100%)] text-white text-[16px] font-medium py-2 px-14 rounded-xl mt-2 inline-block">
                {imageFile ? "Change file" : "Select file"}
                <input
                  id="image_path"
                  type="file"
                  className="hidden"
                  onChange={handleFileUpload(setImageFile)}
                  required
                />
              </label>
            </div>

            <Button
              className="mt-auto ml-auto bg-[linear-gradient(90deg,#9333EA_0%,#6366F1_100%)] text-white text-[20px] font-normal px-10 py-6 rounded-lg"
              onClick={() => setStep(2)}
            >
              Next
            </Button>
        </>
      )}


      {step === 2 && (
          <>
            <h1 className="absolute top-8 left-14 text-[36px] font-normal text-black">Select Architecture</h1>

            <div className="flex justify-between items-start w-full mt-28">
            <div className="relative inline-block bg-[linear-gradient(90deg,#9333EA_0%,#6366F1_100%)] p-[1px] rounded-[13px] ml-8">
              <label className="flex items-center justify-center cursor-pointer bg-white text-black text-[16px] font-medium px-7 py-2 rounded-xl">
                Upload custom file
                <Input
                  type="file"
                  accept=".py"
                  className="hidden"
                  disabled={!!modelName} 
                  onChange={(e) => {
                    if (e.target.files?.length) {
                      setCustomModelFile(e.target.files[0]);
                      setModelName(""); 
                      setArchitectureSelected(false);
                    }
                  }}
                />
              </label>
            </div>
          
            <div className="relative mr-1 focus:outline-none"> 
            <DropdownMenu.Root open={open} onOpenChange={setOpen}>
              <DropdownMenu.Trigger asChild>
                <button
                  className="flex justify-between w-[12rem] items-center p-2 border rounded-xl shadow-md text-[#737373] text-[16px] font-normal bg-white mr-8"
                  onClick={() => setOpen((prev) => !prev)} 
                >
                  {modelName ? modelName : "Select an architecture"}
                  {open ? (
                    <ChevronUp className="h-4 w-4 text-[#737373]" />
                  ) : (
                    <ChevronDown className="h-4 w-4 text-[#737373]" />
                  )}
                </button>
              </DropdownMenu.Trigger>
              <DropdownMenu.Portal>
                <DropdownMenu.Content
                  align="start"
                  side="bottom"
                  sideOffset={6} 
                  className="w-[13rem] max-h-[15rem] p-2 bg-white shadow-lg rounded-md z-50 text-black overflow-y-auto"
                  forceMount
                >
                  <div className="relative px-2 pb-2">
                    <input
                      type="text"
                      placeholder="Find an architecture..."
                      value={search}
                      onChange={(e) => setSearch(e.target.value)}
                      className="w-full px-2 py-2 text-[#737373] text-[16px] font-normal border-b focus:outline-none focus:border-[#4918B2]"
                    />
                  </div>
                  <DropdownMenu.RadioGroup
                    value={modelName}
                    onValueChange={(value) => {
                      setModelName(value);
                      setArchitectureSelected(value !== "None");
                      setCustomModelFile(null); 
                    }}
                  >
                    <DropdownMenu.RadioItem
                      value=""
                      onSelect={(e) => e.preventDefault()} 
                      className={cn(
                        "flex justify-between items-center px-3 py-2 text-[16px] font-normal cursor-pointer rounded-md transition",
                        "hover:bg-[#F6F2FF] hover:border hover:border-[#4918B2]",
                        modelName === "" ? "bg-[#F6F2FF] border border-[#4918B2] text-[#1A1A1A] font-bold " : "text-[#484848]"
                      )}
                    >
                      None
                      {!modelName && <Check className="w-4 h-4 text-[#4918B2]" />}
                    </DropdownMenu.RadioItem>

                    {architectures
                      .filter((arch) => arch.toLowerCase().includes(search.toLowerCase()))
                      .map((architecture) => (
                        <DropdownMenu.RadioItem
                          key={architecture}
                          value={architecture}
                          onSelect={(e) => e.preventDefault()} 
                          className={cn(
                            "flex justify-between items-center px-3 py-2 text-[16px] font-normal cursor-pointer rounded-md transition",
                            "hover:bg-[#F6F2FF] hover:border hover:border-[#4918B2]",
                            modelName === architecture ? "bg-[#F6F2FF] border border-[#4918B2] text-[#1A1A1A] font-bold " : "text-[#484848]"
                          )}
                        >
                          {architecture}
                          {modelName === architecture && <Check className="w-4 h-4 text-[#4918B2]" />}
                        </DropdownMenu.RadioItem>
                      ))}
                  </DropdownMenu.RadioGroup>
                </DropdownMenu.Content>
              </DropdownMenu.Portal>
            </DropdownMenu.Root>
            </div>
      
            </div>

            {customModelFile && (
              <p className="bg-[linear-gradient(90deg,#9333EA_0%,#6366F1_100%)] bg-clip-text text-transparent text-[18px] font-normal mt-2">{customModelFile.name}</p>
            )}

            <div className="flex justify-between items-center mt-auto w-full">
              <Button
                className="mt-6 bg-[linear-gradient(90deg,#9333EA_0%,#6366F1_100%)] text-white text-[20px] font-normal px-10 py-6 rounded-lg"
                onClick={() => setStep(1)}
              >
                Back
              </Button>
              <Button
                className="mt-6 bg-[linear-gradient(90deg,#9333EA_0%,#6366F1_100%)] text-white text-[20px] font-normal px-10 py-6 rounded-lg"
                onClick={() => setStep(3)}
              >
                Next
              </Button>
            </div>
        </>
      )}

      {step === 3 && (
        <>
          <h1 className="absolute top-8 left-14 text-[36px] font-normal text-black">Select Weight</h1>

          <div className="flex justify-between items-start w-full mt-28">
            <div className="relative inline-block bg-[linear-gradient(90deg,#9333EA_0%,#6366F1_100%)] p-[1px] rounded-[13px] ml-8">
              <label className="block bg-white text-black text-[16px] font-medium px-6 py-2 rounded-xl cursor-pointer">
                Upload custom file
                <Input
                  type="file"
                  accept=".pt, .pth, .tar"
                  className="hidden"
                  disabled={weightsSelected} // Disabled when Default is selected
                  onChange={handleFileUpload(setWeightsFile, () => setWeightsSelected(false))}
                />
              </label>
            </div>

            {/* Default Button */}
            <div className="relative inline-block bg-[linear-gradient(90deg,#9333EA_0%,#6366F1_100%)] p-[1px] rounded-[13px] mr-8">
              <Button
                variant="outline"
                className={`w-full px-16 py-4 rounded-xl 
                  ${weightsSelected ? "bg-[#F6F2FF] text-black text-[16px] font-medium" : "bg-white text-black text-[16px] font-medium"}`}
                onClick={() => {
                  setWeightsSelected((prev) => !prev); // Toggle Default selection
                  if (!weightsSelected) {
                    setWeightsFile(null); // Clear file when selecting Default
                  }
                }}
              >
                Default
              </Button>
            </div>
          </div>

          {weightsFile && <p className="bg-[linear-gradient(90deg,#9333EA_0%,#6366F1_100%)] bg-clip-text text-transparent text-[18px] font-normal mt-2 text-center">{weightsFile.name} uploaded</p>}

          <div className="mt-8">
          <Label htmlFor="class_labels_csv" className="text-black text-[16px] font-medium">Class Labels CSV</Label>
          <Input
            id="class_labels_csv"
            type="file"
            className="block w-full text-sm text-gray-500 border border-gray-300 rounded-lg cursor-pointer"
            onChange={(e) => {
              if (e.target.files?.length) {
                setClassLabelsFile(e.target.files[0]); // Save uploaded file
              }
            }}
            required
          />
          </div>

          <div className="flex justify-between items-center mt-auto w-full">
            <Button
              className="bg-[linear-gradient(90deg,#9333EA_0%,#6366F1_100%)] text-white text-[20px] font-normal px-10 py-6 rounded-lg"
              onClick={() => setStep(2)}
            >
              Back
            </Button>

            <SubmitButton
              className="bg-[linear-gradient(90deg,#9333EA_0%,#6366F1_100%)] text-white text-[20px] font-normal px-10 py-6 rounded-lg"
              pendingText="Processing..."
              onClick={async () => {
                setIsLoading(true);
                await handleSubmit(); // Call backend and wait for response
                setIsLoading(false); 

                // Navigate only if visualizationUrl is set (response received)
                if (visualizationUrl) {
                  setStep(4);
                }
              }}
              disabled={isLoading} // Disable button while request is processing
            >
              Done
            </SubmitButton>
          </div>
        </>
      )}

      {step === 4 && (
        <>
            <h2 className="absolute top-8 left-14 text-[36px] font-normal text-black">Generated Visualization</h2>
            {visualizationUrl ? (
              <>
                <div className="flex flex-col justify-center items-center gap-4 mt-28">
                  <img
                    src={visualizationUrl}
                    alt="Grad-CAM Output"
                    className="max-w-[35vh] max-h-[35vh] border rounded shadow-lg"
                  />

                  <div className="text-sm text-foreground text-center">
                    {predictionInfo?.split("\n").map((line, idx) => (
                      <p key={idx}>{line}</p>
                    ))}
                  </div>
                </div>
              </>
            ) : (
              <p className="text-gray-500 mt-4">Processing visualization...</p>
            )}

            <div className="mt-auto mr-auto">
              <Button
                className="bg-[linear-gradient(90deg,#9333EA_0%,#6366F1_100%)] text-white text-[20px] font-normal px-10 py-6 rounded-lg"
                onClick={() => setStep(3)}
              >
                Back
              </Button>
            </div>
        </>
      )}
      
      </div>
    </div>
  );
}
