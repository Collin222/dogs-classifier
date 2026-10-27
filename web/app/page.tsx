"use client";

import { cn } from "@/utils";
import {
	IconAlertTriangle,
	IconAlertTriangleFilled,
  IconExternalLink,
} from "@tabler/icons-react";
import Image from "next/image";
import { useMemo, useRef, useState } from "react";

const ENCODED_W = 224;
const ENCODED_H = 224;
const CHANNEL_SIZE = ENCODED_W * ENCODED_H;
const REPO_URL = "https://github.com/Collin222/dogs-classifier";

const DOGS = ["Charlotte", "Ivy", "Leah"];

const parseImage = async (image: File) => {
	const map = await createImageBitmap(image);

	const canvas = document.createElement("canvas");
	const ctx = canvas.getContext("2d");
	if (!ctx) throw new Error("Failed to get canvas context");
	canvas.width = ENCODED_W;
	canvas.height = ENCODED_H;

	ctx.drawImage(map, 0, 0, ENCODED_W, ENCODED_H);

	const { data } = ctx.getImageData(0, 0, ENCODED_W, ENCODED_H);

  const pixels = [];
  let j = 0;
	for (let i = 0; i < data.length; i += 4) {
		const r = data[i];
		const g = data[i + 1];
		const b = data[i + 2];
    pixels[0 * CHANNEL_SIZE + j] = r;
    pixels[1 * CHANNEL_SIZE + j] = g;
    pixels[2 * CHANNEL_SIZE + j] = b;
    j++;
  }

	if (pixels.length !== ENCODED_W * ENCODED_H * 3)
		throw new Error(
			`Failed to parse image to correct size, got : ${pixels.length.toLocaleString()}`,
		);
	return pixels;
};

interface PredictResponse {
	scores: number[];
	predictions: number[];
}

export default function Home() {
	const [image, setImage] = useState<File>();
	const previewUrl = useMemo(
		() => (image ? URL.createObjectURL(image) : null),
		[image],
	);
	const [loading, setLoading] = useState(false);
	const [error, setError] = useState<string | null>(null);
	const [results, setResults] = useState<PredictResponse | null>();

	const imageInputRef = useRef<HTMLInputElement>(null);

	const predict = async (pixels: number[]): Promise<PredictResponse> => {
		if (!process.env.NEXT_PUBLIC_API_URL) throw new Error("NEXT_PUBLIC_API_URL is not set");
		const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/predict`, {
			headers: {
				"Content-Type": "application/json",
			},
			method: "POST",
			body: JSON.stringify({ pixels }),
		}).catch(e => {
      if (e) console.error(e);
      throw e;
    });
		if (!res.ok)
			throw new Error(
				`Failed to predict: ${res.status} ${res.statusText}`,
			);

		const data = (await res.json()) as PredictResponse;
		return data;
	};

	const handleImage = (file: File) => {
		setError(null);
		setResults(null);
		setLoading(true);

		parseImage(file)
			.then(async (pixels) => {
				const res = await predict(pixels);
				setResults(res);
				setLoading(false);
			})
			.catch((error: Error) => {
				setError(error.message);
				setLoading(false);
				console.error(error);
			});
	};

	return (
		<div className="min-h-screen flex flex-col justify-center items-center px-[20vw] py-25">
			<h1 className="mb-2 text-lg text-center">Dogs CNN Classifier</h1>
			<p className="mb-4 text-gray-300 text-center">
				VGG11 CNN implementation in C++.{" "}
				<a href={REPO_URL} target="_blank" className="underline">
          <IconExternalLink className="inlineicon hover:underline " size="1.1rem" />
					See the source code
				</a>
				.
			</p>

			<button
				onClick={() => {
					imageInputRef.current?.click();
				}}
        className="cursor-pointer"
			>
				Upload Image
			</button>
			<input
				ref={imageInputRef}
				type="file"
				accept="image/*"
				onChange={(e) => {
					const file = e.target.files?.[0];
					if (!file) return;
					setImage(file);
					handleImage(file);
				}}
				multiple={false}
				className="hidden"
			/>

			{error != null && (
				<>
					<h5 className="mt-4">
						<IconAlertTriangleFilled
							className="inlineicon text-red-500"
							size="1.1rem"
						/>{" "}
						Something went wrong
					</h5>
					<p className="text-sm text-gray-300 mt-1">{error}</p>
				</>
			)}

			{previewUrl != null && error == null && (
				<div className="flex flex-nowrap gap-12 mt-12">
					<Image
						src={previewUrl}
						alt="image"
						width={ENCODED_W}
						height={ENCODED_H}
					/>

					<div className="flex flex-col gap-2">
						<h4>Results</h4>
						{loading || results == null ? (
							<div>Loading...</div>
						) : (
							results.scores.map((score, index) => {
								const dog = DOGS[index]!;
								const high = results.predictions[index] === 1;
								return (
									<p
										className={cn(
											high ? "text-teal-500" : null,
										)}
									>
										{dog}: {(score * 100).toFixed(2)}%
									</p>
								);
							})
						)}
					</div>
				</div>
			)}
		</div>
	);
}
