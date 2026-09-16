import * as React from "react";
import { cn } from "@/lib/utils";

const badgeVariants = {
  default:
    "border-transparent bg-primary text-primary-foreground hover:bg-primary/80",
  secondary:
    "border-transparent bg-secondary text-secondary-foreground hover:bg-secondary/80",
  destructive:
    "border-transparent bg-destructive text-destructive-foreground hover:bg-destructive/80",
  outline: "text-foreground",
  green:
    "border-transparent bg-green-400/40 text-secondary-foreground hover:bg-green-600/30 cursor-pointer",
  yellow:
    "border-transparent bg-yellow-500/40 text-secondary-foreground hover:bg-yellow-600/30 cursor-pointer",
  red: "border-transparent bg-red-500/40 text-secondary-foreground hover:bg-red-600/30 cursor-pointer",
  blue: "border-transparent bg-blue-500/40 text-secondary-foreground hover:bg-blue-600/30 cursor-pointer",
};

function Badge({ className, variant = "default", ...props }) {
  const variantClasses = badgeVariants[variant] || badgeVariants.default;

  return (
    <div
      className={cn(
        "inline-flex items-center rounded-full border px-2.5 py-1 text-xs font-semibold transition-colors focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2",
        variantClasses,
        className
      )}
      {...props}
    />
  );
}

export { Badge };
