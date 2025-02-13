using GameRecommender.Models;
using Microsoft.EntityFrameworkCore;

namespace GameRecommender.Data;

public class PostgresHandler : DbContext, IDatabaseHandler
{
    public DbSet<User> Users { get; set; }
    public DbSet<UserToSteamId> UserToSteamIds { get; set; }
    public DbSet<UserGame> UserGames { get; set; }
    public DbSet<AppIdToName> AppIdToNames { get; set; }

    public PostgresHandler(DbContextOptions<PostgresHandler> options) : base(options) {}

    public async Task<User> RegisterUser(User user)
    {
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

    public async Task AddGameToUserLibrary(Guid userId, string appId, double playtime, bool? opinion = null)
    {
        var data = new UserGame(userId, appId, playtime, opinion);
        await UserGames.AddAsync(data);
        await SaveChangesAsync();
    }

    public async Task<List<GameData>> GetUserGames(Guid userId)
    {
        var tasks = UserGames.Where(ug => ug.UserId == userId).AsEnumerable().Select(async ug =>
        {
            return new GameData(
                ug.AppId,
                await AppIdToNames.Where(g => g.AppId == ug.AppId).Select(g => g.Name).FirstOrDefaultAsync() ?? "",
                ug.Playtime);
        }).ToList();

        List<GameData> result = (await Task.WhenAll(tasks)).ToList();
        return result;
    }
}