import type { Metadata } from "next";
import { Poppins } from "next/font/google";
import "./globals.css";

const sans = Poppins({
  subsets: ["latin"],
  variable: "--font-sans",
  weight: ["300", "400", "500", "600", "700"],
});

export const metadata: Metadata = {
  title: "Bias AI System",
  description: "AI fairness audit dashboard",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={sans.variable} suppressHydrationWarning>
        {children}
      </body>
    </html>
  );
}
