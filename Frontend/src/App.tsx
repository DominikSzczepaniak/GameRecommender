import { ThemeProvider } from '@/components/theme-provider';
import { Home } from 'lucide-react';
import { Settings } from 'lucide-react';
import { Route, Routes } from 'react-router-dom';
import Navbar from './components/Navbar';
import Login from './pages/Login';
import Register from './pages/Register';

function App() {
  return (
    <ThemeProvider defaultTheme='dark' storageKey='vite-ui-theme'>
      <div>
        <Navbar />
        <Routes>
          <Route path='/' element={<Home />} />
          <Route path='/login' element={<Login />} />
          <Route path='/register' element={<Register />} />
          <Route path='/settings' element={<Settings />} />
        </Routes>
      </div>
    </ThemeProvider>
  );
}

export default App;
