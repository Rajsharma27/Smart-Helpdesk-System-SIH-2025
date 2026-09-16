import { Toaster } from "@/components/ui/sonner";
import "./globals.css";
import { UserProvider } from "@/context/UserContext";

export const metadata = {
  title: "Ticketing App",
  description: "A ticketing app built with Next.js",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className="font-sans antialiased">
        <UserProvider>{children}</UserProvider>
        <Toaster />
      </body>
    </html>
  );
}
