import { Button } from '@/components/ui/button';
import { Card, CardHeader } from '@/components/ui/card';
import { Skeleton } from '@/components/ui/skeleton';
import { getGamePhoto } from '@/helpers/gamePhotos';
import { API_SERVER } from '@/settings';
import { errorHandler } from '@/utilities/error';
import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';

interface GameGalleryProps {
  appIds: string[];
  maxSelections?: number;
}

type GameDto = {
  AppId: string,
  Opinion: boolean,
};

const GameGallery: React.FC<GameGalleryProps> = ({ appIds, maxSelections = 5 }) => {
  const navigate = useNavigate();
  const [gamesChecked, setGamesChecked] = useState(false);
  const [gamePhotos, setGamePhotos] = useState<Record<string, string | null>>({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedGames, setSelectedGames] = useState<string[]>([]);
  const [selectionLimitReached, setSelectionLimitReached] = useState(false);

  const handleProceed = async () => {
    try {
      const games: GameDto[] = [];
      for(const game in selectedGames) {
        games.push({ AppId: game, Opinion: true });
      }
      const response = await fetch(`${API_SERVER}/Game/addGame`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authentication': `Bearer ${localStorage.getItem('token')}`,
        },
        body: JSON.stringify({
          user: {
            Id: localStorage.getItem('userId'),
            Username: localStorage.getItem('username'),
            Email: localStorage.getItem('email'),
            Password: '',
          },
          gameDto: selectedGames,
        }),
      });

      if (!response.ok) {
        return errorHandler('Faield to save liked games');
      }

      console.log('Correctly saved liked games');

      window.location.href = '/';
    } catch (error) {
      return errorHandler('Error submitting games: '+error);
    }
  };

  const handleGameClick = (appId: string) => {
    if (selectedGames.includes(appId)) {
      setSelectedGames(selectedGames.filter((id) => id !== appId));
      setSelectionLimitReached(false);
    } else if (selectedGames.length < maxSelections) {
      setSelectedGames([...selectedGames, appId]);
    } else {
      setSelectionLimitReached(true);
    }
  };

  const isSelected = (appId: string) => selectedGames.includes(appId);

  useEffect(() => {
    const gameChosen = async () => {
      try {
        const response = await fetch(`${API_SERVER}/User/gamesChosen`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authentication': `Bearer ${localStorage.getItem('token')}`,
          },
          body: JSON.stringify({
            user: {
              Id: localStorage.getItem('userId'),
              Username: localStorage.getItem('username'),
              Email: localStorage.getItem('email'),
              Password: '',
            },
          }),
        });

        if (!response.ok) {
          throw new Error('Request failed');
        }

        const data = await response.json();

        if (data) {
          navigate('/');
        } else {
          setGamesChecked(true);
        }
      } catch (error) {
        console.error('Error choosing games:', error);
        setGamesChecked(true);
      }
    };

    gameChosen();
  }, [navigate]);

  useEffect(() => {
    const fetchGamePhotos = async () => {
      try {
        const promises = appIds.map(async (appId) => {
          const imageUrl = getGamePhoto(appId);
          const response = await fetch(imageUrl);
          if (!response.ok) {
            console.warn(`Image not found for appID: ${appId}`);
            return { appId, imageUrl: null };
          }
          return { appId, imageUrl };
        });

        const results = await Promise.all(promises);
        const photos: Record<string, string | null> = {};
        results.forEach(({ appId, imageUrl }) => {
          photos[appId] = imageUrl;
        });
        setGamePhotos(photos);
      } catch (err) {
        console.error('Error fetching game photos:', err);
        setError('Error fetching game photos. Please try again later.');
      } finally {
        setLoading(false);
      }
    };

    fetchGamePhotos();
  }, [appIds]);

  if (loading) {
    return (
      <div className='grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4'>
        {appIds.map((appId) => (
          <Card key={appId}>
            <CardHeader>
              <Skeleton className='h-48 w-full' />
            </CardHeader>
          </Card>
        ))}
      </div>
    );
  }

  if (error) {
    return (
      <div className='text-center text-red-500'>
        {error}
      </div>
    );
  }

  if (!gamesChecked) {
    return (
      <div>
        Loading...
      </div>
    );
  }

  return (
    <div>
      {selectionLimitReached && (
        <p className='text-red-500 mb-2'>
          You can only select up to {maxSelections} games.
        </p>
      )}
      <div className='grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4'>
        {appIds.map((appId) => (
          <Card
            key={appId}
            className={`cursor-pointer ${isSelected(appId) ? 'border-4 border-blue-500' : ''}`}
            onClick={() => handleGameClick(appId)}>
            <CardHeader>
              {gamePhotos[appId]
                ? (
                  <img
                    src={gamePhotos[appId]!}
                    alt={`Game ${appId} Header`}
                    className='w-full h-auto object-cover rounded-t-lg'
                    loading='lazy' />
                )
                : (
                  <div className='bg-gray-200 h-48 flex items-center justify-center rounded-t-lg'>
                    <p className='text-gray-500'>
                      No Image Available
                    </p>
                  </div>
                )}
            </CardHeader>
          </Card>
        ))}
      </div>
      <div className='mt-4 flex flex-col items-center'>
        {/* Centered content */}
        <p>
          Selected Games: {selectedGames.length} / {maxSelections}
        </p>
        {selectedGames.length > 0 && (
          <p>
            Selected App IDs: {selectedGames.join(', ')}
          </p>
        )}
        <Button onClick={handleProceed} disabled={selectedGames.length === 0} className='mt-2'>
          {/* Proceed button */}
          Proceed
        </Button>
      </div>
    </div>
  );
};

export default GameGallery;
