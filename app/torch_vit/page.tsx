"use client"

import React, { useState } from "react"
import MetadataCard from "@/components/metadata-card"
import BreadcrumbSteps from "@/components/BreadcrumbSteps"
import NameStep from "@/components/NameStep"
import ImageUploadStep from "@/components/ImageUploadStep"

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

const steps = ["Name", "Image Upload", "Architecture", "Weights", "Model Visualization"]

export default function TorchVIT() {
    const [step, setStep] = useState(1)
    const [fileName, setFileName] = useState("")
    const [modelName, setModelName] = useState("")
    const [weightsFile, setWeightsFile] = useState<File | null>(null)
    const [imageFile, setImageFile] = useState<File | null>(null)
    const [classLabelsFile, setClassLabelsFile] = useState<File | null>(null)
    const [visualizationUrl, setVisualizationUrl] = useState<string | null>(null)
    const [predictionInfo, setPredictionInfo] = useState<string | null>(null)
    const [isLoading, setIsLoading] = useState(false)
    const [error, setError] = useState<string | null>(null)
    const [architectureSelected, setArchitectureSelected] = useState(false)
    const [customModelFile, setCustomModelFile] = useState<File | null>(null)
    const [weightsSelected, setWeightsSelected] = useState(false)
    const [dragActive, setDragActive] = useState(false)
    const [search, setSearch] = React.useState("")
    const [open, setOpen] = useState(false)

    const handleFileUpload =
        (
            setter: React.Dispatch<React.SetStateAction<File | null>>,
            setFlag?: React.Dispatch<React.SetStateAction<boolean>>,
        ) =>
        (event: React.ChangeEvent<HTMLInputElement>) => {
            if (event.target.files) {
                setter(event.target.files[0])
                if (setFlag) setFlag(true)
            }
        }

    const handleSubmit = async () => {
        setIsLoading(true)
        setError(null)

        if ((!modelName && !customModelFile) || !imageFile || !classLabelsFile) {
            setError("Error: Model selection, image file, and class labels CSV are required.")
            setIsLoading(false)
            return
        }

        const formData = new FormData()

        if (customModelFile && weightsFile) {
            formData.append("custom_model_file", customModelFile)
            formData.append("custom_weights_file", weightsFile)
        } else if (modelName) {
            formData.append("model_name", modelName)
        }

        formData.append("image_path", imageFile as File)
        formData.append("class_labels_csv", classLabelsFile as File)

        if (!customModelFile && !weightsSelected && weightsFile) {
            formData.append("weights_path", weightsFile)
        }

        try {
            const response = await fetch("/api/pytorch/heatmap", { method: "POST", body: formData })
            if (!response.ok) {
                throw new Error("Failed to generate Grad-CAM visualization.")
            }
            const result = await response.json()
            setVisualizationUrl(`data:image/png;base64,${result.image}`)
            setPredictionInfo(`Predicted Class: ${result.predicted_class} (${result.predicted_probability})`)
        } catch (error) {
            setError("Error processing request. Please try again.")
        } finally {
            setIsLoading(false)
        }
    }

    return (
        <div className="w-full px-4 pt-28 pb-8">
            {step === 1 && (
                <NameStep fileName={fileName} setFileName={setFileName} setStep={setStep} />
            )}

            {step === 2 && (
                <div className="flex flex-col md:flex-row gap-6 w-full max-w-6xl mx-auto">
                    <div className="w-full md:w-1/3">
                        <MetadataCard
                            name={fileName}
                            status="In Progress"
                            id="41603"
                            createdDate="2/26/2025"
                            institution="UC Davis"
                            creator="rbihani"
                        />
                    </div>

                    <div className="w-full md:w-2/3">
                        <div className="bg-[#210B2C] rounded-xl shadow-[0_0_6px_4px_rgba(141,57,235,0.5)] p-6 sm:p-8 md:p-10 flex flex-col">
                            <BreadcrumbSteps steps={steps} currentStepIndex={step-1} />
                            <div className="mt-6">
                                <ImageUploadStep
                                    imageFile={imageFile}
                                    setImageFile={setImageFile}
                                    setStep={setStep}
                                />
                            </div>
                        </div>
                    </div>
                </div>
            )}
        </div>
    )
}
