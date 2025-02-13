'use client';

import React, { useState } from "react";
import { SubmitButton } from "@/components/submit-button";
import { Input } from "@/components/ui/input";
import { DropdownMenu } from "@/components/ui/dropdown-menu";
import { Label } from "@/components/ui/label";
import image_upload from "@/components/static/image-upload.svg";
import Image from "next/image";

export default function TorchVIT()
{
    const [modelName, setModelName] = useState("");
    const [weightsFile, setWeightsFile] = useState<File | null>(null);
    const [imageFile, setImageFile] = useState<File | null>(null);
    const [classLabelsFile, setClassLabelsFile] = useState<File | null>(null);
    const [numClasses, setNumClasses] = useState('1000');
    const [visualizationUrl, setVisualizationUrl] = useState<string | null>(null);
    const [predictionInfo, setPredictionInfo] = useState<string | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const handleFileUpload = (setter: React.Dispatch<React.SetStateAction<File | null>>) => (event: React.ChangeEvent<HTMLInputElement>) => 
    {
        if (event.target.files) {
            setter(event.target.files[0]);
        }
    };

    const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
        event.preventDefault();
        setIsLoading(true);
        setError(null);
    
        if (!modelName || !weightsFile || !imageFile || !classLabelsFile || !numClasses) {
            setError("Error: Model name, weights file, image file, class labels CSV, and number of classes are required.");
            setIsLoading(false);
            return;
        }
    
        const formData = new FormData();
        formData.append("model_name", modelName);
        formData.append('weights_file', weightsFile as File);
        formData.append("image_path", imageFile as File);
        formData.append("class_labels_csv", classLabelsFile as File);    
        formData.append("num_classes", numClasses);

        try {
            const response = await fetch("/api/pytorch/heatmap_vit", { method: "POST", body: formData });
    
            if (!response.ok) {
                // throw new Error(`${response.json()}`)
                throw new Error("Failed to generate attention rollout visualization.");
            }
    
            const result = await response.json();
            
            setVisualizationUrl(`data:image/png;base64,${result.image}`);
            setPredictionInfo(`Predicted Class: ${result.predicted_class}`);
        } catch (error) {
            setError(`${error}`);
        } finally {
            setIsLoading(false);
        }
    };

    return(
        
        <div className="container mx-auto">
            <h1 className="pb-7 text-center text-2xl font-bold">Vision Transformer Attention Rollout Visualization</h1>

            {isLoading && <p className="text-center font-bold">Loading...</p>}
            {error && <p className="text-center font-bold text-red-500">{error}</p>}

            <div className="container flex flex-row">
                <form onSubmit={handleSubmit} className="space-y-4 container flex flex-row">
                    <div className='container mx-auto w-2/5 p-4 h-96 flex flex-col '>
                        <h5 className='p-2 text-lg font-bold'>Input sample below</h5>
                        <div className='border-2 rounded-2xl border-[#D5714B] bg-[#F5DED3]/20'>
                            <div className="flex flex-row justify-center pt-4"> 
                                <Image
                                    src={image_upload}
                                    alt=''
                                    height={120}
                                    width={120} />
                            </div>  
                            <h3 className="pt-4 px-4 text-center text-s font-bold text-[#D5714B]">Drag and drop an image below</h3>
                            <h4 className="text-center text-s text-[#D5714B]"> or</h4>
                            <div className="justify-center mx-auto">
                                <Input className="border-none bg-transparent rounded-2xl"
                                    id="image_path"
                                    type="file"
                                    accept="image/*"
                                    onChange={handleFileUpload(setImageFile)}
                                    required
                                ></Input>
                            </div>
                        </div>
                    </div>
                    <div className="container">
                        <h3 className="text-xl pt-2 font-bold text-left">Fill in details</h3>
                        <div className="p-2">
                            <Label htmlFor="model_name">Model Name<span className="text-red-500">*</span></Label>
                            <Input
                                id="model_name"
                                type="text"
                                value={modelName}
                                onChange={(e) => setModelName(e.target.value)}
                                placeholder="e.g., vit_b_16, vit_b_32"
                                required
                              />
                        </div>
                        <div className="p-2">
                            <Label htmlFor="num_classes">Number of Classes <span className="text-red-500">*</span></Label>
                            <Input
                                id="num_classes"
                                type="text"
                                onChange={(e) => setNumClasses(e.target.value)}
                                placeholder="eg. 2, 1000"
                                required
                            />
                        </div>
                        <div className="p-2">
                            <Label htmlFor="weights_path">Weights File<span className="text-red-500">*</span></Label>
                            <Input
                                id="weights_path"
                                type="file"
                                accept=".pth,.tar"
                                onChange={handleFileUpload(setWeightsFile)}
                            />
                        </div>
                        <div className="p-2">
                            <Label htmlFor="class_labels_csv">Class Labels CSV <span className="text-red-500">*</span></Label>
                            <Input
                                id="class_labels_csv"
                                type="file"
                                accept=".csv"
                                onChange={handleFileUpload(setClassLabelsFile)}
                                required
                            />
                        </div>
                        <div>
                            <SubmitButton className="mt-2 bg-[#D5714B]" pendingText="Processing...">Run Inference</SubmitButton>
                        </div>
                    </div>
                </form>
            </div>

            {visualizationUrl && (
            <div className="mt-8">
                <h2 className="text-xl font-medium">Attention Rollout Output</h2>
                <img
                    src={visualizationUrl}
                    alt="Attention Rollout Output"
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
    )
}