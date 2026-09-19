import type { Metadata } from 'next';
import { Geist, Geist_Mono } from 'next/font/google';
import './globals.css';

const geistSans = Geist({
  variable: '--font-geist-sans',
  subsets: ['latin'],
});

const geistMono = Geist_Mono({
  variable: '--font-geist-mono',
  subsets: ['latin'],
});

export const metadata: Metadata = {
  metadataBase: new URL(process.env.NEXT_PUBLIC_SITE_URL || 'http://localhost:3000'),
  title: 'FluentFrame · 视频英语学习工作台',
  description: '把英文视频自动变成逐句跟随、双语翻译与挖空练习。',
  openGraph: {
    title: 'FluentFrame · 把视频变成你的英语课堂',
    description: '逐句跟随、双语翻译、挖空练习与单词彩色高亮。',
    images: ['/og.png'],
    locale: 'zh_CN',
    type: 'website',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'FluentFrame · 把视频变成你的英语课堂',
    description: '逐句跟随、双语翻译、挖空练习与单词彩色高亮。',
    images: ['/og.png'],
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="zh-CN">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        {children}
      </body>
    </html>
  );
}
