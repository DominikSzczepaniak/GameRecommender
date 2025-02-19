import { LoginRegisterForm } from '@/components/LoginRegisterForm';

const Register = () => {
  return (
    <div>
      <div className='flex items-center justify-center h-screen'>
        <LoginRegisterForm isLogin={false} />
      </div>
    </div>
  );
};

export default Register;
