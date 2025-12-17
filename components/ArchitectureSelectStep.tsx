"use client"

import type React from "react"
import { useState, useRef, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Check, ChevronDown, ChevronUp } from "lucide-react"
import { cn } from "@/lib/utils"

interface ArchitectureSelectStepProps {
  modelName: string
  setModelName: (name: string) => void
  weightsFile: File | null
  setWeightsFile: (file: File | null) => void
  classLabelsFile: File | null
  setClassLabelsFile: (file: File | null) => void
  customModelFile: File | null
  setCustomModelFile: (file: File | null) => void
  setStep: (step: number) => void
  numClasses: number
  setNumClasses: (num: number) => void
  imageSize: number
  setImageSize: (size: number) => void
  architectureType : string
  setArchitectureType: (architecture: string) => void
}

const architectures = [
  "resnet18",
  "resnet34",
  "resnet50",
  "resnet101",
  "resnet152",
  "vgg11",
  "vgg13",
  "vgg16",
  "vgg19",
  "alexnet",
  "inception_v3",
  "densenet121",
  "densenet169",
  "densenet201",
  "densenet161",
  "mobilenet_v2",
  "shufflenet_v2_x0_5",
  "shufflenet_v2_x1_0",
]

const vitModels = [
    "vit_base_mci_224",
    "vit_base_patch8_224",
    "vit_base_patch14_dinov2",
    "vit_base_patch14_reg4_dinov2",
    "vit_base_patch16_18x2_224",
    "vit_base_patch16_224",
    "vit_base_patch16_224_miil",
    "vit_base_patch16_384",
    "vit_base_patch16_clip_224",
    "vit_base_patch16_clip_384",
    "vit_base_patch16_clip_quickgelu_224",
    "vit_base_patch16_gap_224",
    "vit_base_patch16_plus_240",
    "vit_base_patch16_plus_clip_240",
    "vit_base_patch16_reg4_gap_256",
    "vit_base_patch16_rope_reg1_gap_256",
    "vit_base_patch16_rpn_224",
    "vit_base_patch16_siglip_224",
    "vit_base_patch16_siglip_256",
    "vit_base_patch16_siglip_384",
    "vit_base_patch16_siglip_512",
    "vit_base_patch16_siglip_gap_224",
    "vit_base_patch16_siglip_gap_256",
    "vit_base_patch16_siglip_gap_384",
    "vit_base_patch16_siglip_gap_512",
    "vit_base_patch16_xp_224",
    "vit_base_patch32_224",
    "vit_base_patch32_384",
    "vit_base_patch32_clip_224",
    "vit_base_patch32_clip_256",
    "vit_base_patch32_clip_384",
    "vit_base_patch32_clip_448",
    "vit_base_patch32_clip_quickgelu_224",
    "vit_base_patch32_plus_256",
    "vit_base_r26_s32_224",
    "vit_base_r50_s16_224",
    "vit_base_r50_s16_384",
    "vit_base_resnet26d_224",
    "vit_base_resnet50d_224",
    "vit_betwixt_patch16_gap_256",
    "vit_betwixt_patch16_reg1_gap_256",
    "vit_betwixt_patch16_reg4_gap_256",
    "vit_betwixt_patch16_reg4_gap_384",
    "vit_betwixt_patch16_rope_reg4_gap_256",
    "vit_betwixt_patch32_clip_224",
    "vit_giant_patch14_224",
    "vit_giant_patch14_clip_224",
    "vit_giant_patch14_dinov2",
    "vit_giant_patch14_reg4_dinov2",
    "vit_giant_patch16_gap_224",
    "vit_gigantic_patch14_224",
    "vit_gigantic_patch14_clip_224",
    "vit_gigantic_patch14_clip_quickgelu_224",
    "vit_huge_patch14_224",
    "vit_huge_patch14_clip_224",
    "vit_huge_patch14_clip_336",
    "vit_huge_patch14_clip_378",
    "vit_huge_patch14_clip_quickgelu_224",
    "vit_huge_patch14_clip_quickgelu_378",
    "vit_huge_patch14_gap_224",
    "vit_huge_patch14_xp_224",
    "vit_huge_patch16_gap_448",
    "vit_intern300m_patch14_448",
    "vit_large_patch14_224",
    "vit_large_patch14_clip_224",
    "vit_large_patch14_clip_336",
    "vit_large_patch14_clip_quickgelu_224",
    "vit_large_patch14_clip_quickgelu_336",
    "vit_large_patch14_dinov2",
    "vit_large_patch14_reg4_dinov2",
    "vit_large_patch14_xp_224",
    "vit_large_patch16_224",
    "vit_large_patch16_384",
    "vit_large_patch16_siglip_256",
    "vit_large_patch16_siglip_384",
    "vit_large_patch16_siglip_gap_256",
    "vit_large_patch16_siglip_gap_384",
    "vit_large_patch32_224",
    "vit_large_patch32_384",
    "vit_large_r50_s32_224",
    "vit_large_r50_s32_384",
    "vit_little_patch16_reg1_gap_256",
    "vit_little_patch16_reg4_gap_256",
    "vit_medium_patch16_clip_224",
    "vit_medium_patch16_gap_240",
    "vit_medium_patch16_gap_256",
    "vit_medium_patch16_gap_384",
    "vit_medium_patch16_reg1_gap_256",
    "vit_medium_patch16_reg4_gap_256",
    "vit_medium_patch16_rope_reg1_gap_256",
    "vit_medium_patch32_clip_224",
    "vit_mediumd_patch16_reg4_gap_256",
    "vit_mediumd_patch16_reg4_gap_384",
    "vit_mediumd_patch16_rope_reg1_gap_256",
    "vit_pwee_patch16_reg1_gap_256",
    "vit_relpos_base_patch16_224",
    "vit_relpos_base_patch16_cls_224",
    "vit_relpos_base_patch16_clsgap_224",
    "vit_relpos_base_patch16_plus_240",
    "vit_relpos_base_patch16_rpn_224",
    "vit_relpos_base_patch32_plus_rpn_256",
    "vit_relpos_medium_patch16_224",
    "vit_relpos_medium_patch16_cls_224",
    "vit_relpos_medium_patch16_rpn_224",
    "vit_relpos_small_patch16_224",
    "vit_relpos_small_patch16_rpn_224",
    "vit_small_patch8_224",
    "vit_small_patch14_dinov2",
    "vit_small_patch14_reg4_dinov2",
    "vit_small_patch16_18x2_224",
    "vit_small_patch16_36x1_224",
    "vit_small_patch16_224",
    "vit_small_patch16_384",
    "vit_small_patch32_224",
    "vit_small_patch32_384",
    "vit_small_r26_s32_224",
    "vit_small_r26_s32_384",
    "vit_small_resnet26d_224",
    "vit_small_resnet50d_s16_224",
    "vit_so150m2_patch16_reg1_gap_256",
    "vit_so150m_patch16_reg4_gap_256",
    "vit_so150m_patch16_reg4_gap_384",
    "vit_so150m_patch16_reg4_map_256",
    "vit_so400m_patch14_siglip_224",
    "vit_so400m_patch14_siglip_378",
    "vit_so400m_patch14_siglip_384",
    "vit_so400m_patch14_siglip_gap_224",
    "vit_so400m_patch14_siglip_gap_378",
    "vit_so400m_patch14_siglip_gap_384",
    "vit_so400m_patch14_siglip_gap_448",
    "vit_so400m_patch14_siglip_gap_896",
    "vit_so400m_patch16_siglip_256",
    "vit_so400m_patch16_siglip_gap_256",
    "vit_srelpos_medium_patch16_224",
    "vit_srelpos_small_patch16_224",
    "vit_tiny_patch16_224",
    "vit_tiny_patch16_384",
    "vit_tiny_r_s16_p8_224",
    "vit_tiny_r_s16_p8_384",
    "vit_wee_patch16_reg1_gap_256",
    "vit_xsmall_patch16_clip_224",
    "deit_base_distilled_patch16_224",
    "deit_base_distilled_patch16_384",
    "deit_base_patch16_224",
    "deit_base_patch16_384",
    "deit_small_distilled_patch16_224",
    "deit_small_patch16_224",
    "deit_tiny_distilled_patch16_224",
    "deit_tiny_patch16_224",
]

const ArchitectureSelectStep: React.FC<ArchitectureSelectStepProps> = ({
  modelName,
  setModelName,
  weightsFile,
  setWeightsFile,
  classLabelsFile,
  setClassLabelsFile,
  customModelFile,
  setCustomModelFile,
  setStep,
  numClasses,
  setNumClasses,
  imageSize,
  setImageSize,
  architectureType,
  setArchitectureType,
}) => {
//   const [architectureType, setArchitectureType] = useState<"cnn" | "vit">("cnn")
  const [search, setSearch] = useState("")
  const [dropdownOpen, setDropdownOpen] = useState(false)
  const dropdownRef = useRef<HTMLDivElement>(null)

  // Handle clicks outside the dropdown to close it
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setDropdownOpen(false)
      }
    }

    document.addEventListener("mousedown", handleClickOutside)
    return () => {
      document.removeEventListener("mousedown", handleClickOutside)
    }
  }, [])

  const handleFileUpload =
    (setter: React.Dispatch<React.SetStateAction<File | null>>) => (event: React.ChangeEvent<HTMLInputElement>) => {
      if (event.target.files && event.target.files[0]) {
        setter(event.target.files[0])
      }
    }

  const handleTabChange = (type: "cnn" | "vit") => {
    setArchitectureType(type)
    setModelName("")
    setCustomModelFile(null)
    setClassLabelsFile(null)
    setWeightsFile(null)
    setNumClasses(0)
    setImageSize(0)
    setSearch("")
    setDropdownOpen(false)
  }

  const filteredArchitectures =
    architectureType === "cnn"
      ? architectures.filter((arch) => arch.toLowerCase().includes(search.toLowerCase()))
      : vitModels.filter((model) => model.toLowerCase().includes(search.toLowerCase()))

  return (
    <div className="flex flex-col h-full w-full">
      <h1 className="text-2xl font-light mb-6">Select Architecture</h1>

      {/* Architecture Type Selection Cards */}
      <div className="flex flex-col md:flex-row gap-6 mb-8">
        {/* CNN Option */}
        <div
          className={`flex-1 p-6 border rounded-lg cursor-pointer transition-all ${
            architectureType === "cnn"
              ? "border-purple-500 bg-[#2D1139]"
              : "border-gray-700 bg-[#210B2C] hover:border-purple-400"
          }`}
          onClick={() => handleTabChange("cnn")}
        >
          <div className="flex items-start">
            <div>
              <h3 className="text-white text-lg font-medium">Convolutional Neural Networks</h3>
              <p className="text-gray-400 text-sm mt-1">
                Traditional CNN architectures like ResNet, VGG, DenseNet, and Inception for image classification.
              </p>
            </div>
          </div>
        </div>

        {/* Vision Transformer Option */}
        <div
          className={`flex-1 p-6 border rounded-lg cursor-pointer transition-all ${
            architectureType === "vit"
              ? "border-purple-500 bg-[#2D1139]"
              : "border-gray-700 bg-[#210B2C] hover:border-purple-400"
          }`}
          onClick={() => handleTabChange("vit")}
        >
          <div className="flex items-start">
            <div>
              <h3 className="text-white text-lg font-medium">Vision Transformers</h3>
              <p className="text-gray-400 text-sm mt-1">
                Developed by Google, ViTs are modern transformer-based architectures designed for computer vision tasks.
              </p>
            </div>
          </div>
        </div>
      </div>

      {architectureType === "cnn" && (
        <div className="space-y-6">
		<p className="text-center font-medium">Select an existing supported architecture or upload your own.</p>
          <div className="mb-6">
            <div className="flex flex-col md:flex-row gap-4">
              <div className="w-full md:w-1/2">
                <Label htmlFor="cnn_model" className="text-white text-base font-medium mb-2 block">
                  Convolutional Neural Network Model
                </Label>
                <div className="relative w-full" ref={dropdownRef}>
                  <button
                    id="cnn_model"
                    className="flex justify-between w-full items-center p-2 border rounded-lg shadow-md text-gray-300 text-white font-normal bg-[#210B2C] border-purple-600"
                    onClick={() => setDropdownOpen(!dropdownOpen)}
                  >
                    {modelName ? modelName : "Select architecture"}
                    {dropdownOpen ? (
                      <ChevronUp className="h-4 w-4 text-gray-300" />
                    ) : (
                      <ChevronDown className="h-4 w-4 text-gray-300" />
                    )}
                  </button>

                  {dropdownOpen && (
                    <div className="absolute z-10 mt-1 w-full max-h-60 overflow-auto bg-[#210B2C] border border-gray-700 rounded-md shadow-lg">
                      <div className="p-2">
                        <input
                          type="text"
                          placeholder="Find architecture..."
                          value={search}
                          onChange={(e) => setSearch(e.target.value)}
                          className="w-full px-2 py-2 text-gray-300 text-sm font-normal bg-[#210B2C] border-b border-gray-700 focus:outline-none focus:border-purple-500"
                          onClick={(e) => e.stopPropagation()}
                        />
                      </div>
                      <div className="py-1">
                        <button
                          className={cn(
                            "flex justify-between items-center w-full px-3 py-2 text-sm font-normal text-left rounded-md transition",
                            "hover:bg-[#2D1139] hover:border hover:border-purple-500",
                            !modelName
                              ? "bg-[#2D1139] border border-purple-500 text-white font-medium"
                              : "text-gray-300",
                          )}
                          onClick={() => {
                            setModelName("")
                            setCustomModelFile(null)
                            setDropdownOpen(false)
                          }}
                        >
                          None
                          {!modelName && <Check className="w-4 h-4 text-purple-500" />}
                        </button>

                        {filteredArchitectures.map((architecture) => (
                          <button
                            key={architecture}
                            className={cn(
                              "flex justify-between items-center w-full px-3 py-2 text-sm font-normal text-left rounded-md transition",
                              "hover:bg-[#2D1139] hover:border hover:border-purple-500",
                              modelName === architecture
                                ? "bg-[#2D1139] border border-purple-500 text-white font-medium"
                                : "text-gray-300",
                            )}
                            onClick={() => {
                              setModelName(architecture)
                              setCustomModelFile(null)
                              setDropdownOpen(false)
                            }}
                          >
                            {architecture}
                            {modelName === architecture && <Check className="w-4 h-4 text-purple-500" />}
                          </button>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
                {modelName && <p className="text-purple-400 text-sm mt-1">Selected: {modelName}</p>}
              </div>

              <div className="w-full md:w-1/2">
                <Label htmlFor="custom_model" className="text-white text-base font-medium mb-2 block">
                  Custom Model Upload
                </Label>
                <label className="flex items-center justify-center whitespace-nowrap text-sm sm:text-white cursor-pointer bg-[#210B2C] text-white font-normal px-4 sm:px-6 py-2 rounded-lg hover:bg-[#2D1139] transition border border-purple-600 w-full h-10">
                  Upload custom model
                  <Input
                    id="custom_model"
                    type="file"
                    accept=".py"
                    className="hidden"
                    disabled={!!modelName}
                    onChange={(e) => {
                      if (e.target.files?.length) {
                        setCustomModelFile(e.target.files[0])
                        setModelName("")
                      }
                    }}
                  />
                </label>
                {customModelFile && <p className="text-purple-400 text-sm mt-1">Selected: {customModelFile.name}</p>}
              </div>
            </div>
          </div>

          <div>
            <Label htmlFor="class_labels_csv" className="text-white text-base font-medium mb-2 block">
              Class Labels CSV File
            </Label>
            <Input
              id="class_labels_csv"
              type="file"
              accept=".csv"
              className="block h-fit w-full text-sm text-gray-300 bg-[#210B2C] border border-purple-100 border-opacity-25 rounded-lg cursor-pointer file:mr-4 file:py-2 file:px-4 file:rounded-md file:border-0 file:text-sm file:font-medium file:bg-purple-600 file:text-white hover:file:bg-purple-700"
              onChange={handleFileUpload(setClassLabelsFile)}
              required
            />
            {classLabelsFile && <p className="text-purple-400 text-sm mt-1">Selected: {classLabelsFile.name}</p>}
          </div>
        </div>
      )}

      {architectureType === "vit" && (
        <div className="space-y-6">
          <div className="mb-6">
            <Label htmlFor="vit_model" className="text-white text-base font-medium mb-2 block">
              Vision Transformer Model
            </Label>
            <div className="relative w-full" ref={dropdownRef}>
              <button
                id="vit_model"
                className="flex justify-between w-full items-center p-2 border rounded-lg shadow-md text-gray-300 text-white font-normal bg-[#210B2C] border-purple-600"
                onClick={() => setDropdownOpen(!dropdownOpen)}
              >
                {modelName ? modelName : "Select architecture"}
                {dropdownOpen ? (
                  <ChevronUp className="h-4 w-4 text-gray-300" />
                ) : (
                  <ChevronDown className="h-4 w-4 text-gray-300" />
                )}
              </button>

              {dropdownOpen && (
                <div className="absolute z-10 mt-1 w-full max-h-60 overflow-auto bg-[#210B2C] border border-gray-700 rounded-md shadow-lg">
                  <div className="p-2">
                    <input
                      type="text"
                      placeholder="Find model..."
                      value={search}
                      onChange={(e) => setSearch(e.target.value)}
                      className="w-full px-2 py-2 text-gray-300 text-sm font-normal bg-[#210B2C] border-b border-gray-700 focus:outline-none focus:border-purple-500"
                      onClick={(e) => e.stopPropagation()}
                    />
                  </div>
                  <div className="py-1">
                    <button
                      className={cn(
                        "flex justify-between items-center w-full px-3 py-2 text-sm font-normal text-left rounded-md transition",
                        "hover:bg-[#2D1139] hover:border hover:border-purple-500",
                        !modelName ? "bg-[#2D1139] border border-purple-500 text-white font-medium" : "text-gray-300",
                      )}
                      onClick={() => {
                        setModelName("")
                        setDropdownOpen(false)
                      }}
                    >
                      None
                      {!modelName && <Check className="w-4 h-4 text-purple-500" />}
                    </button>

                    {filteredArchitectures.map((model) => (
                      <button
                        key={model}
                        className={cn(
                          "flex justify-between items-center w-full px-3 py-2 text-sm font-normal text-left rounded-md transition",
                          "hover:bg-[#2D1139] hover:border hover:border-purple-500",
                          modelName === model
                            ? "bg-[#2D1139] border border-purple-500 text-white font-medium"
                            : "text-gray-300",
                        )}
                        onClick={() => {
                          setModelName(model)
                          setDropdownOpen(false)
                        }}
                      >
                        {model}
                        {modelName === model && <Check className="w-4 h-4 text-purple-500" />}
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </div>
            {modelName && <p className="text-purple-400 text-sm mt-1">Selected: {modelName}</p>}
          </div>

          <div className="mb-6">
            <Label htmlFor="class_labels_csv" className="text-white text-base font-medium mb-2 block">
              Class Labels CSV
            </Label>
            <Input
              id="class_labels_csv"
              type="file"
              accept=".csv"
              className="block h-fit w-full text-sm text-gray-300 bg-[#210B2C] border border-purple-100 border-opacity-25 rounded-lg cursor-pointer file:mr-4 file:py-2 file:px-4 file:rounded-md file:border-0 file:text-sm file:font-medium file:bg-purple-600 file:text-white hover:file:bg-purple-700"
              onChange={handleFileUpload(setClassLabelsFile)}
              required
            />
            {classLabelsFile && <p className="text-purple-400 text-sm mt-1">Selected: {classLabelsFile.name}</p>}
          </div>
		
			<div>
            <Label htmlFor="num_classes" className="text-white text-base font-medium mb-2 block">
              Number of Classes
            </Label>
            <Input
              id="num_classes"
              type="number"
              min="1"
              value={numClasses || ""}
              onChange={(e) => setNumClasses(Number.parseInt(e.target.value) || 0)}
              className="bg-[#210B2C] border border-purple-100 border-opacity-25 text-white rounded-lg focus:ring-purple-500 focus:border-purple-500 h-10"
              placeholder="e.g., 2"
              required
            />
            <p className="text-gray-400 text-xs mt-1">Number of classes for classification. This should be the length of your CSV file.</p>
          </div>

          <div>
            <Label htmlFor="image_size" className="text-white text-base font-medium mb-2 block">
              Image Size
            </Label>
            <Input
              id="image_size"
              type="number"
              min="1"
              value={imageSize || ""}
              onChange={(e) => setImageSize(Number.parseInt(e.target.value) || 0)}
              className="bg-[#210B2C] border border-purple-100 border-opacity-25 text-white rounded-lg focus:ring-purple-500 focus:border-purple-500 h-10"
              placeholder="e.g., 224"
              required
            />
            <p className="text-gray-400 text-xs mt-1">Input image size for the model (e.g., 224 for 224x224)</p>
          </div>
        </div>
      )}

      <div className="flex justify-between mt-auto pt-6">
        <Button
          className="bg-gradient-to-r from-purple-600 to-indigo-500 text-white text-base font-normal px-8 py-2 rounded-lg hover:opacity-90 transition"
          onClick={() => setStep(2)}
        >
          Back
        </Button>
        <Button
          className="bg-gradient-to-r from-purple-600 to-indigo-500 text-white text-base font-normal px-8 py-2 rounded-lg hover:opacity-90 transition"
          onClick={() => setStep(4)}
        >
          Next
        </Button>
      </div>
    </div>
  )
}

export default ArchitectureSelectStep
