"use client";

import AppSidebar from "@/components/ui/menu/AppSidebar";
import Navbar from "@/components/ui/menu/Navbar";
import { SidebarProvider } from "@/components/ui/sidebar";
import { TabProvider } from "@/context/TabContext";
import { useUser } from "@/context/UserContext";

function layout({ children }) {
  const { isLoading } = useUser();
  return (
    <div>
      <TabProvider>
        <SidebarProvider>
          {!isLoading && <AppSidebar />}
          <div className="w-full">
            {!isLoading && <Navbar />}
            <main className="py-3 px-16">{children}</main>
          </div>
        </SidebarProvider>
      </TabProvider>
    </div>
  );
}

export default layout;
