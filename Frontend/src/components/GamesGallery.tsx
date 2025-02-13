import { Button } from '@/components/ui/button';
import { Card, CardHeader } from '@/components/ui/card';
import { Skeleton } from '@/components/ui/skeleton';
import React, { useEffect, useState } from 'react';

interface GameGalleryProps {
  appIds: number[];
  maxSelections?: number;
}

const GameGallery: React.FC<GameGalleryProps> = ({ appIds, maxSelections = 5 }) => {
  const [gamePhotos, setGamePhotos] = useState<Record<number, string | null>>({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedGames, setSelectedGames] = useState<number[]>([]);
  const [selectionLimitReached, setSelectionLimitReached] = useState(false);
  const userId = localStorage.getItem("userId") as string as unknown as number;

  const getGamePhoto = (appId: number): string => {
    return `https://shared.fastly.steamstatic.com/store_item_assets/steam/apps/${appId}/header.jpg`;
  };

  const handleProceed = async (userId: number) => {
    try {
      // Placeholder API call (replace with your actual API call)
      console.log(`Sending selected games for userId ${userId} to API:`, selectedGames);

      // const response = await fetch('/api/submitGames', {  // Example API endpoint
      //   method: 'POST',
      //   headers: {
      //     'Content-Type': 'application/json',
      //   },
      //   body: JSON.stringify({ userId, selectedGames }),
      // });

      // if (!response.ok) {
      //   throw new Error(`API request failed with status ${response.status}`);
      // }

      // const data = await response.json();
      // console.log("API response:", data);

      window.location.href = '/';
    } catch (error) {
      console.error('Error submitting games:', error);
      // Handle error (e.g., display an error message to the user)
    }
  };

  const handleGameClick = (appId: number) => {
    if (selectedGames.includes(appId)) {
      setSelectedGames(selectedGames.filter((id) => id !== appId));
      setSelectionLimitReached(false);
    } else if (selectedGames.length < maxSelections) {
      setSelectedGames([...selectedGames, appId]);
    } else {
      setSelectionLimitReached(true);
    }
  };

  const checkIfAlreadyDone = (userId: number) => {
    //ask api if userId already done gameGallery
    return null;
  }

  const isSelected = (appId: number) => selectedGames.includes(appId);

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
        const photos: Record<number, string | null> = {};
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
        <Button onClick={() => handleProceed(userId)} disabled={selectedGames.length === 0} className='mt-2'>
          {/* Proceed button */}
          Proceed
        </Button>
      </div>
    </div>
  );
};

export default GameGallery;
