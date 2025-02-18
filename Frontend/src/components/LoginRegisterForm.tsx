import { Button } from '@/components/ui/button';
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Switch } from '@/components/ui/switch';
import { AuthContext } from '@/context/AuthProvider';
import { cn } from '@/lib/utils';
import { User } from '@/models/User';
import { API_SERVER } from '@/settings';
import { errorHandler } from '@/utilities/error';
import { useContext, useState } from 'react';
import { useNavigate } from 'react-router-dom';

type FormProps = React.ComponentProps<typeof Card> & {
  isLogin: boolean,
};

const EMAIL_REGEX = new RegExp('^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+.[a-zA-Z]{2,}$');

const validator = (email: string) => {
  return EMAIL_REGEX.test(email);
};

export function LoginRegisterForm({ className, isLogin, ...props }: FormProps) {
  const [username, setUsername] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [password2, setPassword2] = useState('');
  const [message, setMessage] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const { login } = useContext(AuthContext);

  const navigate = useNavigate();

  const checkGamesChosen = async () => {
    try {
      const response = await fetch(`${API_SERVER}/User/gamesChosen`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
          'Authentication': `Bearer ${localStorage.getItem('token')}`,
        },
      });

      if (!response.ok) {
        throw new Error('Request failed');
      }

      const data = await response.json();
      return data;
    } catch (error) {
      console.error('Error choosing games:', error);
    }
  };

  const handleLogin = async (username: string, password: string) => {
    try {
      const response = await fetch(`${API_SERVER}/User/login`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ username, password }),
      });

      if (!response.ok) {
        return errorHandler('Login failed');
      }

      const data = await response.json();
      await localStorage.setItem('token', data.token);
      await localStorage.setItem('userId', data.user['id']);
      await localStorage.setItem('username', data.user['username']);
      await localStorage.setItem('email', data.user['email']);
      login();
      setMessage('');
      if (await checkGamesChosen()) {
        return navigate('/gamesGallery');
      } else {
        return navigate('/');
      }
    } catch (error) {
      console.error('Error:', error);
      return null;
    }
  };

  const handleRegister = async (username: string, email: string, password: string) => {
    const user: User = { Id: crypto.randomUUID(), Username: username, Email: email, Password: password };
    try {
      const response = await fetch(`${API_SERVER}/User/register`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(user),
      });

      if (!response.ok) {
        throw new Error('Login failed');
      }
      return; //TODO show toast here and redirect
    } catch (error) {
      console.error('Error:', error);
      return null; //TODO toast
    }
  };

  const handleLoginRegister = (username: string, password: string, isLogin: boolean, email: string) => {
    if (!username || !password) {
      setMessage('One of the inputs is empty!');
    } else if (!isLogin && !validator(email)) {
      setMessage('Wrong email format! Follow: example@gmail.com');
    } else {
      if (isLogin) {
        handleLogin(username, password);
      } else {
        handleRegister(username, email, password);
        setMessage('');
        navigate('/login');
      }
    }
  };

  return (
    <Card className={cn('w-[450px]', className)} {...props}>
      <CardHeader>
        <CardTitle>
          {isLogin ? 'Login' : 'Register'}
        </CardTitle>
      </CardHeader>
      <CardContent className='grid gap-4'>
        <div className=' flex items-center space-x-4 rounded-md border p-4'>
          <div className='flex-1 space-y-1'>
            <p>
              Username
            </p>
            <Input
              placeholder='Type in your username'
              value={username}
              onChange={(e) => {
                setUsername(e.target.value);
              }} />
            {!isLogin && (
              <div className='mt-2'>
                <p>
                  Email
                </p>
                <Input
                  placeholder='Type in your email'
                  value={email}
                  onChange={(e) => {
                    setEmail(e.target.value);
                  }} />
              </div>
            )}
            <p>
              Password
            </p>
            <Input
              type={showPassword ? 'text' : 'password'}
              placeholder='Type in your password'
              value={password}
              onChange={(e) => {
                setPassword(e.target.value);
              }} />
            {!isLogin && (
              <div className='mt-2'>
                <p>
                  Repeat Password
                </p>
                <Input
                  className='mt-2'
                  type={showPassword ? 'text' : 'password'}
                  placeholder='Retype your password'
                  value={password2}
                  onChange={(e) => {
                    setPassword2(e.target.value);
                  }} />
              </div>
            )}
            {isLogin && (
              <div>
                <div className='flex items-center space-x-2 text-sm text-muted-foreground mt-2'>
                  <Switch
                    onClick={() => {
                      setShowPassword(!showPassword);
                    }} />
                  <p>
                    Show password
                  </p>
                </div>
                <p className='text-sm text-muted-foreground mt-2'>
                  <a className='mt-2 hover:text-blue-700 hover:underline transition-colors' href='/register'>
                    Not registered yet?
                  </a>
                </p>
              </div>
            )}
            <div className='text-mm text-red'>
              {message}
            </div>
          </div>
        </div>
      </CardContent>
      <CardFooter>
        <Button
          className='w-full'
          onClick={() => {
            handleLoginRegister(username, password, isLogin, email);
          }}>
          {isLogin ? 'Login' : 'Register'}
        </Button>
      </CardFooter>
    </Card>
  );
}
