import viteLogo from '/vite.svg'
import { ThemeProvider } from '@/components/theme-provider'
import { useState } from 'react'
import { Route, Routes } from 'react-router-dom'
import { Home } from 'lucide-react'
import { Settings } from 'lucide-react'
import Navbar from './components/Navbar'
import Register from './pages/Register'
import Login from './pages/Login'

function App() {
  const [count, setCount] = useState(0);

  return (
    <ThemeProvider defaultTheme="dark" storageKey="vite-ui-theme">
      <div>
        <Navbar />
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/login" element={<Login />} />
          <Route path="/register" element={<Register />} />
          <Route path="/settings" element={<Settings />} />
        </Routes>
      </div>
    </ThemeProvider>
  );
}

export default App;
