import { clsx } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs) {
  return twMerge(clsx(inputs));
}

export const getToken = () => {
  if (typeof window === "undefined") return null;
  return window.localStorage.getItem("token");
};
