"use client";

import { useUser } from "@/context/UserContext";
import { useRouter } from "next/navigation";
import { BounceLoader } from "react-spinners";
import { useEffect } from "react";

export default function Home() {
  const router = useRouter();
  useEffect(() => {
    router.push("/auth");
  });
  return (
    <div className="h-screen w-screen flex items-center justify-center">
      <BounceLoader color="#4A90E2" />
    </div>
  );
}
