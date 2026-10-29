import { cn } from "@/utils";
import { Tooltip, TooltipContent, TooltipTrigger } from "./ui/tooltip";

export default function TooltipLink({
	href,
	children,
	className,
}: {
	href: string;
	children: React.ReactNode;
	className?: string;
}) {
	return (
		<Tooltip>
			<TooltipContent>{href}</TooltipContent>
			<TooltipTrigger>
				<a
					href={href}
					target="_blank"
					className={cn(
						"underline md:no-underline hover:underline text-accent-foreground",
						className,
					)}
				>
					{children}
				</a>
			</TooltipTrigger>
		</Tooltip>
	);
}
