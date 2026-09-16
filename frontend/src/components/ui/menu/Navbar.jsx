"use client";

import { Bell, CircleUserRound } from "lucide-react";
import React from "react";
import { useUser } from "@/context/UserContext";

const Navbar = () => {
  const { currentUser } = useUser();
  return (
    <nav className="px-12 h-18.5 border-b border-border flex flex-row-reverse items-center gap-6">
      <div className="p-2 rounded-full bg-muted-foreground/40 cursor-pointer">
        <CircleUserRound size={32} className="text-muted stroke-1" />
      </div>
      <div className="hover:bg-muted p-2 w-fit flex rounded-full cursor-pointer relative">
        <Bell />
        <div className="h-1.5 w-1.5 bg-destructive rounded-full absolute top-0 right-0 translate-x-1/2 -translate-y-1/2" />
      </div>
    </nav>
  );
};

export default Navbar;
