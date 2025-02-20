using GameRecommender.Interfaces.Data;
using GameRecommender.Models;

namespace GameRecommender.Data;

public class PostgresUserHandler(PostgresConnectionPool connectionPool) : PostgresBase(connectionPool), IUserHandler
{
    public async Task RegisterUser(User user)
    {
        using var connection = await GetConnectionAsync();
        var command = connection.CreateCommand();
        command.CommandText = "INSERT INTO Users (Id, Username, Email, Password) VALUES (@Id, @Username, @Email, @Password)";
        command.Parameters.Add(new Npgsql.NpgsqlParameter("@Id", user.Id));
        command.Parameters.Add(new Npgsql.NpgsqlParameter("@Username", user.Username));
        command.Parameters.Add(new Npgsql.NpgsqlParameter("@Email", user.Email));
        command.Parameters.Add(new Npgsql.NpgsqlParameter("@Password", user.Password));

        var existingUser = await UserExists(user.Username);
        if (existingUser)
        {
            throw new ArgumentException("User already exists");
        }

        await command.ExecuteNonQueryAsync();
    }

    private async Task<bool> UserExists(string username)
    {
        using var connection = await GetConnectionAsync();
        var command = connection.CreateCommand();
        command.CommandText = "SELECT 1 FROM Users WHERE Username = @Username";
        command.Parameters.Add(new Npgsql.NpgsqlParameter("@Username", username));

        using var reader = await command.ExecuteReaderAsync();
        return await reader.ReadAsync(); // Returns true if a row is found, false otherwise
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
                Email = reader.GetString(reader.GetOrdinal("Email")),
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
}