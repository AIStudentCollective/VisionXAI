"use client"

import type React from "react"
import { useState } from "react"
import { Button } from "@/components/ui/button"

interface ImageUploadProps {
  imageFile: File | null
  setImageFile: (file: File | null) => void
  setStep: (step: number) => void
}

const ImageUploadStep: React.FC<ImageUploadProps> = ({ imageFile, setImageFile, setStep }) => {
  const [dragActive, setDragActive] = useState(false)

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files[0]) {
      setImageFile(event.target.files[0])
    }
  }

  // Drag-and-drop handlers
  const handleDragOver = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault()
    setDragActive(true)
  }

  const handleDragLeave = () => {
    setDragActive(false)
  }

  const handleDrop = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault()
    setDragActive(false)
    if (event.dataTransfer.files && event.dataTransfer.files.length > 0) {
      setImageFile(event.dataTransfer.files[0])
    }
  }

  return (
    <div className="flex flex-col h-full w-full">
      <h1 className="text-2xl font-light mb-6">Upload image for visualization</h1>

      <div
        className={`flex-1 border-2 border-dashed rounded-lg flex flex-col items-center justify-center text-center p-6 transition-all duration-200 ${
          dragActive ? "bg-[#161918] border-purple-500" : "bg-[#161918] border-gray-500"
        }`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        <img src="/images/mage_image-upload.svg" alt="Upload Icon" className="w-16 h-16 mb-4" />

        {imageFile ? (
          <p className="text-white text-lg font-light mb-2">{imageFile.name}</p>
        ) : (
          <>
            <p className="text-white text-lg font-light mb-2">Choose a file or drag & drop it here</p>
            <p className="text-gray-400 text-sm mb-4">JPEG, PNG, PDG formats, up to 50MB</p>
          </>
        )}

        <label className="cursor-pointer bg-white text-[#210B2C] text-base font-medium py-2 px-8 rounded-lg hover:bg-gray-100 transition">
          {imageFile ? "Change file" : "Browse File"}
          <input
            id="image_path"
            type="file"
            className="hidden"
            onChange={handleFileUpload}
            accept="image/jpeg,image/png,image/pdg"
          />
        </label>
      </div>

      <div className="flex justify-end mt-6">
        <Button
          className="bg-gradient-to-r from-purple-600 to-indigo-500 text-white text-base font-normal px-8 py-2 rounded-lg hover:opacity-90 transition"
          disabled={!imageFile}
          onClick={() => setStep(3)}
        >
          Next
        </Button>
      </div>
    </div>
  )
}

export default ImageUploadStep;
