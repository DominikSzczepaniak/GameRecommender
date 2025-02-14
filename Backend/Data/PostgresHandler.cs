using GameRecommender.Models;
using Microsoft.EntityFrameworkCore;

namespace GameRecommender.Data;

public class PostgresHandler : DbContext, IDatabaseHandler
{
    public DbSet<User> Users { get; set; }
    public DbSet<UserToSteamId> UserToSteamIds { get; set; }
    public DbSet<UserGameDao> UserGames { get; set; }
    public DbSet<AppIdToName> AppIdToNames { get; set; }

    public PostgresHandler(DbContextOptions<PostgresHandler> options) : base(options) {}

    public async Task<User> RegisterUser(User user)
    {
        if (Users.Any(u => u.Id == user.Id))
        {
            throw new ArgumentException("User already exists");
        }
        await Users.AddAsync(user);
        await SaveChangesAsync();
        return user;
    }

    public async Task<User?> LoginByUsername(string username, string password)
    {
        return await Users.FirstOrDefaultAsync(u => u.Username == username && u.Password == password);
    }

    public async Task<User> UpdateUser(User user)
    {
        Users.Update(user);
        await SaveChangesAsync();
        return user;
    }

    public async Task<bool> DeleteUser(User user)
    {
        // if (!Users.Any(u => u.Id == user.Id))
        // {
        //     return false; //could be removed because if SaveChangesAsync() == 0 then we didnt delete anyone hence user didn't exist
        // }
        Users.Remove(user);
        return await SaveChangesAsync() > 0;
    }

    public async Task SetUserSteamProfileId(Guid userId, string steamProfileId)
    {
        var existingEntry = await UserToSteamIds.FindAsync(userId);

        if (existingEntry != null)
        {
            existingEntry.SteamId = steamProfileId;
        }
        else
        {
            await UserToSteamIds.AddAsync(new UserToSteamId { UserId = userId, SteamId = steamProfileId });
        }

        await SaveChangesAsync();
    }

    public async Task<string> GetUserSteamId(Guid userId)
    {
        var result = await UserToSteamIds.Where(us => us.UserId == userId).Select(us => us.SteamId)
            .FirstOrDefaultAsync();
        if (result == null)
        {
            throw new ArgumentException("User doesnt exist or SteamID not saved");
        }
        return result;
    }

    public async Task AddGameToUserLibrary(UserGameDao userGameDao)
    {
        await UserGames.AddAsync(userGameDao);
        await SaveChangesAsync();
    }

    public async Task AddOpinionForUserAndGame(UserGameDao userGameDao)
    {
        var exists = UserGames.Any(p => p.UserId == userGameDao.UserId && p.AppId == userGameDao.AppId);
        if (exists)
        {
            await UserGames.Where(ug => ug.UserId == userGameDao.UserId && ug.AppId == userGameDao.AppId).ExecuteUpdateAsync(set => set.SetProperty(ug => ug.Opinion, userGameDao.Opinion));
            await SaveChangesAsync();
        }
        await AddGameToUserLibrary(userGameDao);
    }

    public Task AddAppIdToNameMapping(string appId, string name)
    {
        AppIdToNames.AddAsync(new AppIdToName(appId, name));
        return SaveChangesAsync();
    }

    public async Task<List<UserGameLogic>> GetUserGames(Guid userId)
    {
        var tasks = UserGames.Where(ug => ug.UserId == userId).AsEnumerable().Select(async ug =>
        {
            return new UserGameLogic(
                ug.AppId,
                ug.Playtime,
                ug.Opinion,
                await AppIdToNames.Where(g => g.AppId == ug.AppId).Select(g => g.Name).FirstOrDefaultAsync() ?? "");
        }).ToList();

        List<UserGameLogic> result = (await Task.WhenAll(tasks)).ToList();
        return result;
    }
}