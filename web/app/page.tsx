"use client";

import TooltipLink from "@/components/TooltipLink";
import {
	Card,
	CardContent,
	CardDescription,
	CardHeader,
	CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Spinner } from "@/components/ui/spinner";
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

const DOGS = ["Charlotte", "Ivy", "Leah"];
const TOTAL_PARAMETERS = 159619;

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
	const [ms, setMs] = useState<number | null>(null);

	const predict = async (pixels: number[]): Promise<PredictResponse> => {
		if (!process.env.NEXT_PUBLIC_API_URL)
			throw new Error("NEXT_PUBLIC_API_URL is not set");
		const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/predict`, {
			headers: {
				"Content-Type": "application/json",
			},
			method: "POST",
			body: JSON.stringify({ pixels }),
		}).catch((e) => {
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
		setMs(null);

		const start = performance.now();
		parseImage(file)
			.then(async (pixels) => {
				const res = await predict(pixels);
				setResults(res);
				setLoading(false);
				const end = performance.now();
				setMs(end - start);
			})
			.catch((error: Error) => {
				setError(error.message);
				setLoading(false);
				console.error(error);
			});
	};

	return (
		<div className="min-h-screen flex flex-col items-center px-[5vw] py-6">
			<Card className="w-full">
				<CardHeader>
					<CardTitle>Dogs CNN Image Classifier</CardTitle>
					<CardDescription>
						VGG11 CNN implementation from scratch in C++.{" "}
						<TooltipLink href="https://github.com/Collin222/dogs-classifier">
							<IconExternalLink
								className="inlineicon hover:underline "
								size="1rem"
							/>
							See the source code
						</TooltipLink>
						. Made my{" "}
						<TooltipLink href="https://github.com/Collin222">
							<IconExternalLink
								className="inlineicon hover:underline "
								size="1rem"
							/>
							Collin
						</TooltipLink>
						. {TOTAL_PARAMETERS.toLocaleString()} parameter model.
					</CardDescription>
				</CardHeader>

				<CardContent>
					<Label htmlFor="image" className="mb-1">
						Select an image to classify
					</Label>
					<Input
						id="image"
						type="file"
						accept="image/*"
						onChange={(e) => {
							const file = e.target.files?.[0];
							if (!file) return;
							setImage(file);
							handleImage(file);
						}}
						multiple={false}
						className="w-fit max-w-[90%] cursor-pointer"
					/>

					{error != null && (
						<Card className="mt-4">
							<CardHeader>
								<CardTitle>
									<IconAlertTriangleFilled
										className="inlineicon text-red-500"
										size="1.1rem"
									/>{" "}
									Something went wrong
								</CardTitle>
								<CardDescription>
									{error}
								</CardDescription>
							</CardHeader>
						</Card>
					)}

					{previewUrl != null && error == null && (
						<Card className="mt-4">
							<CardHeader>
								<CardTitle>
									Results
									{!loading &&
										ms != null &&
										` (${(ms * 1000).toFixed(2)}s)`}
								</CardTitle>
								<CardDescription className="flex flex-row gap-4 flex-wrap gap-y-0.5">
									{loading || results == null ? (
										<div className="flex items-center gap-2">
											<Spinner /> Loading...
										</div>
									) : (
										results.scores.map((score, index) => {
											const dog = DOGS[index]!;
											const high =
												results.predictions[index] ===
												1;
											return (
												<p
													className={cn(
														high
															? "text-teal-500"
															: null,
													)}
												>
													{dog}:{" "}
													{(score * 100).toFixed(2)}%
												</p>
											);
										})
									)}
								</CardDescription>
							</CardHeader>

							<CardContent className="flex flex-nowrap gap-4 md:gap-12">
								<Image
									src={previewUrl}
									alt="image"
									width={ENCODED_W}
									height={ENCODED_H}
									className="rounded-md w-full md:max-w-[400px] w-5/6"
								/>
							</CardContent>
						</Card>
					)}
				</CardContent>
			</Card>
		</div>
	);
}
