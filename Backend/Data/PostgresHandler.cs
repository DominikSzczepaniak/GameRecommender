using GameRecommender.Models;
using Npgsql;

namespace GameRecommender.Data;

public class PostgresHandler(PostgresConnectionPool connectionPool) : IDatabaseHandler, IDisposable
{
    public void Dispose()
    {
        connectionPool.Dispose();
    }

    private async Task<NpgsqlConnection> GetConnectionAsync()
    {
        return await connectionPool.GetConnectionAsync();
    }

    public async Task RegisterUser(User user)
    {
        using var connection = await GetConnectionAsync();
        var command = connection.CreateCommand();
        command.CommandText = "INSERT INTO Users (Id, Username, Email, Password) VALUES (@Id, @Username, @Email, @Password)";
        command.Parameters.Add(new Npgsql.NpgsqlParameter("@Id", user.Id));
        command.Parameters.Add(new Npgsql.NpgsqlParameter("@Username", user.Username));
        command.Parameters.Add(new Npgsql.NpgsqlParameter("@Email", user.Email));
        command.Parameters.Add(new Npgsql.NpgsqlParameter("@Password", user.Password));

        var existingUser = await GetUserByUsername(user.Username);
        if (existingUser != null)
        {
            throw new ArgumentException("User already exists");
        }

        await command.ExecuteNonQueryAsync();
    }

    public async Task<User?> LoginByUsername(string username, string password)
    {
        using var connection = await GetConnectionAsync();
        var command = connection.CreateCommand();
        command.CommandText = "SELECT * FROM Users WHERE Username = @Username AND Password = @Password";
        command.Parameters.Add(new Npgsql.NpgsqlParameter("@Username", username));
        command.Parameters.Add(new Npgsql.NpgsqlParameter("@Password", password));

        using var reader = await command.ExecuteReaderAsync();
        if (await reader.ReadAsync())
        {
            return new User
            {
                Id = reader.GetGuid(reader.GetOrdinal("Id")),
                Username = reader.GetString(reader.GetOrdinal("Username")),
                Password = reader.GetString(reader.GetOrdinal("Password"))
            };
        }

        return null;
    }

    public async Task<User> UpdateUser(User user)
    {
        using var connection = await GetConnectionAsync();
        var command = connection.CreateCommand();
        command.CommandText = "UPDATE Users SET Username = @Username, Password = @Password WHERE Id = @Id";
        command.Parameters.Add(new Npgsql.NpgsqlParameter("@Id", user.Id));
        command.Parameters.Add(new Npgsql.NpgsqlParameter("@Username", user.Username));
        command.Parameters.Add(new Npgsql.NpgsqlParameter("@Password", user.Password));

        await command.ExecuteNonQueryAsync();
        return user;
    }

    public async Task<bool> DeleteUser(User user)
    {
        using var connection = await GetConnectionAsync();
        var command = connection.CreateCommand();
        command.CommandText = "DELETE FROM Users WHERE Id = @Id";
        command.Parameters.Add(new Npgsql.NpgsqlParameter("@Id", user.Id));

        var rowsAffected = await command.ExecuteNonQueryAsync();
        return rowsAffected > 0;
    }

    public async Task SetUserSteamProfileId(Guid userId, string steamProfileId)
    {
        using var connection = await GetConnectionAsync();
        var command = connection.CreateCommand();
        command.CommandText = @"
            INSERT INTO UserToSteamIds (UserId, SteamId)
            VALUES (@UserId, @SteamId)
            ON CONFLICT (UserId) DO UPDATE SET SteamId = EXCLUDED.SteamId";
        command.Parameters.Add(new Npgsql.NpgsqlParameter("@UserId", userId));
        command.Parameters.Add(new Npgsql.NpgsqlParameter("@SteamId", steamProfileId));

        await command.ExecuteNonQueryAsync();
    }

    public async Task<string> GetUserSteamId(Guid userId)
    {
        using var connection = await GetConnectionAsync();
        var command = connection.CreateCommand();
        command.CommandText = "SELECT SteamId FROM UserToSteamIds WHERE UserId = @UserId";
        command.Parameters.Add(new Npgsql.NpgsqlParameter("@UserId", userId));

        var result = await command.ExecuteScalarAsync();
        if (result == null)
        {
            throw new ArgumentException("User doesn't exist or SteamID not saved");
        }

        return result.ToString();
    }

    public async Task AddGameToUserLibrary(UserGameDao userGameDao)
    {
        using var connection = await GetConnectionAsync();
        var command = connection.CreateCommand();
        command.CommandText = "INSERT INTO UserGames (UserId, AppId, Playtime, Opinion) VALUES (@UserId, @AppId, @Playtime, @Opinion)";
        command.Parameters.Add(new Npgsql.NpgsqlParameter("@UserId", userGameDao.UserId));
        command.Parameters.Add(new Npgsql.NpgsqlParameter("@AppId", userGameDao.AppId));
        command.Parameters.Add(new Npgsql.NpgsqlParameter("@Playtime", userGameDao.Playtime));
        command.Parameters.Add(new Npgsql.NpgsqlParameter("@Opinion", userGameDao.Opinion));

        await command.ExecuteNonQueryAsync();
    }

    public async Task AddOpinionForUserAndGame(UserGameDao userGameDao)
    {
        using var connection = await GetConnectionAsync();
        var command = connection.CreateCommand();
        command.CommandText = @"
            UPDATE UserGames 
            SET Opinion = @Opinion 
            WHERE UserId = @UserId AND AppId = @AppId";
        command.Parameters.Add(new Npgsql.NpgsqlParameter("@UserId", userGameDao.UserId));
        command.Parameters.Add(new Npgsql.NpgsqlParameter("@AppId", userGameDao.AppId));
        command.Parameters.Add(new Npgsql.NpgsqlParameter("@Opinion", userGameDao.Opinion));

        var rowsAffected = await command.ExecuteNonQueryAsync();
        if (rowsAffected == 0)
        {
            await AddGameToUserLibrary(userGameDao);
        }
    }

    public async Task AddAppIdToNameMapping(string appId, string name)
    {
        using var connection = await GetConnectionAsync();
        var command = connection.CreateCommand();
        command.CommandText = "INSERT INTO AppIdToNames (AppId, Name) VALUES (@AppId, @Name)";
        command.Parameters.Add(new Npgsql.NpgsqlParameter("@AppId", appId));
        command.Parameters.Add(new Npgsql.NpgsqlParameter("@Name", name));

        await command.ExecuteNonQueryAsync();
    }

    public async Task<List<UserGameLogic>> GetUserGames(Guid userId)
    {
        using var connection = await GetConnectionAsync();
        var command = connection.CreateCommand();
        command.CommandText = @"
            SELECT ug.AppId, ug.Playtime, ug.Opinion, an.Name 
            FROM UserGames ug
            LEFT JOIN AppIdToNames an ON ug.AppId = an.AppId
            WHERE ug.UserId = @UserId";
        command.Parameters.Add(new Npgsql.NpgsqlParameter("@UserId", userId));

        var games = new List<UserGameLogic>();
        using var reader = await command.ExecuteReaderAsync();
        while (await reader.ReadAsync())
        {
            games.Add(new UserGameLogic(
                reader.GetString(reader.GetOrdinal("AppId")),
                reader.GetInt32(reader.GetOrdinal("Playtime")),
                reader.GetBoolean(reader.GetOrdinal("Opinion")),
                reader.IsDBNull(reader.GetOrdinal("Name")) ? "" : reader.GetString(reader.GetOrdinal("Name"))
            ));
        }

        return games;
    }

    private async Task<User?> GetUserByUsername(string username)
    {
        using var connection = await GetConnectionAsync();
        var command = connection.CreateCommand();
        command.CommandText = "SELECT * FROM Users WHERE Username = @Username";
        command.Parameters.Add(new Npgsql.NpgsqlParameter("@Username", username));

        using var reader = await command.ExecuteReaderAsync();
        if (await reader.ReadAsync())
        {
            return new User
            {
                Id = reader.GetGuid(reader.GetOrdinal("Id")),
                Username = reader.GetString(reader.GetOrdinal("Username")),
                Password = reader.GetString(reader.GetOrdinal("Password"))
            };
        }

        return null;
    }
}