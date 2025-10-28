import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'
import Navbar from '../components/Navbar'
import { ThemeProvider } from './contexts/ThemeContext'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'PEARatings',
  description: 'college football power ratings, spread predictions, and historical data analysis',
  keywords: 'college football, power ratings, CFB, analytics, spreads, predictions, PEAR, pear, college baseball, college ratings',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <ThemeProvider>
          <Navbar />
          {children}
          <footer className="bg-slate-900 dark:bg-slate-950 text-white py-8">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
              <p className="text-slate-400 dark:text-slate-500">© 2025 PEARatings</p>
              <p className="text-slate-500 dark:text-slate-600 text-sm mt-2">
                <a href="https://x.com/PEARatings" target="_blank" rel="noopener noreferrer" className="hover:text-slate-300 dark:hover:text-slate-400 transition-colors">
                  @PEARatings
                </a>
              </p>
            </div>
          </footer>
        </ThemeProvider>
      </body>
    </html>
  )
}