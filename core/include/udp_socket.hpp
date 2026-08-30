#pragma once

#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#ifdef _WIN32
    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
    #endif
    #include <winsock2.h>
    #include <ws2tcpip.h>
    #pragma comment(lib, "Ws2_32.lib")
    using socket_t = SOCKET;
    constexpr socket_t INVALID_SOCKET_FD = INVALID_SOCKET;
    constexpr int SOCKET_ERROR_VAL = SOCKET_ERROR;
#else
    #include <arpa/inet.h>
    #include <netinet/in.h>
    #include <sys/socket.h>
    #include <sys/types.h>
    #include <unistd.h>
    using socket_t = int;
    constexpr socket_t INVALID_SOCKET_FD = -1;
    constexpr int SOCKET_ERROR_VAL = -1;
#endif

namespace defense {
namespace telemetry {

/**
 * @brief Cross-Platform Low-Latency Raw UDP Socket Abstraction.
 * Transparently wraps Windows Winsock2 and Linux POSIX BSD Sockets.
 */
class UdpSocket {
public:
    UdpSocket() : sock_(INVALID_SOCKET_FD) {
        init_network_stack();
        sock_ = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
        if (sock_ == INVALID_SOCKET_FD) {
            std::cerr << "[!] Failed to create UDP socket" << std::endl;
        }
    }

    ~UdpSocket() {
        close_socket();
    }

    // Disable copy semantics
    UdpSocket(const UdpSocket&) = delete;
    UdpSocket& operator=(const UdpSocket&) = delete;

    // Enable move semantics
    UdpSocket(UdpSocket&& other) noexcept : sock_(other.sock_) {
        other.sock_ = INVALID_SOCKET_FD;
    }

    UdpSocket& operator=(UdpSocket&& other) noexcept {
        if (this != &other) {
            close_socket();
            sock_ = other.sock_;
            other.sock_ = INVALID_SOCKET_FD;
        }
        return *this;
    }

    /**
     * @brief Binds socket to a local port for receiving UDP telemetry.
     */
    bool bind_port(uint16_t port, const std::string& ip = "0.0.0.0") {
        if (sock_ == INVALID_SOCKET_FD) return false;

        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(port);
        inet_pton(AF_INET, ip.c_str(), &addr.sin_addr);

        int res = ::bind(sock_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr));
        if (res == SOCKET_ERROR_VAL) {
            std::cerr << "[!] UDP Bind failed on " << ip << ":" << port << std::endl;
            return false;
        }
        return true;
    }

    /**
     * @brief Sets read timeout in milliseconds for non-blocking poll.
     */
    bool set_receive_timeout(int timeout_ms) {
        if (sock_ == INVALID_SOCKET_FD) return false;

#ifdef _WIN32
        DWORD tv = static_cast<DWORD>(timeout_ms);
        int res = setsockopt(sock_, SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<const char*>(&tv), sizeof(tv));
#else
        struct timeval tv;
        tv.tv_sec = timeout_ms / 1000;
        tv.tv_usec = (timeout_ms % 1000) * 1000;
        int res = setsockopt(sock_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
#endif
        return res == 0;
    }

    /**
     * @brief Receives raw datagram into buffer.
     * Returns number of bytes received, or -1 on error/timeout.
     */
    int receive_from(void* buffer, size_t max_len, std::string* sender_ip = nullptr, uint16_t* sender_port = nullptr) {
        if (sock_ == INVALID_SOCKET_FD) return -1;

        sockaddr_in client_addr{};
#ifdef _WIN32
        int addr_len = sizeof(client_addr);
#else
        socklen_t addr_len = sizeof(client_addr);
#endif

        int bytes_read = ::recvfrom(
            sock_,
            static_cast<char*>(buffer),
            static_cast<int>(max_len),
            0,
            reinterpret_cast<sockaddr*>(&client_addr),
            &addr_len
        );

        if (bytes_read > 0 && sender_ip) {
            char ip_str[INET_ADDRSTRLEN];
            inet_ntop(AF_INET, &(client_addr.sin_addr), ip_str, INET_ADDRSTRLEN);
            *sender_ip = ip_str;
            if (sender_port) {
                *sender_port = ntohs(client_addr.sin_port);
            }
        }
        return bytes_read;
    }

    /**
     * @brief Sends datagram to destination host and port.
     */
    int send_to(const std::string& host, uint16_t port, const void* data, size_t len) {
        if (sock_ == INVALID_SOCKET_FD) return -1;

        sockaddr_in dest_addr{};
        dest_addr.sin_family = AF_INET;
        dest_addr.sin_port = htons(port);
        inet_pton(AF_INET, host.c_str(), &dest_addr.sin_addr);

        return ::sendto(
            sock_,
            static_cast<const char*>(data),
            static_cast<int>(len),
            0,
            reinterpret_cast<sockaddr*>(&dest_addr),
            sizeof(dest_addr)
        );
    }

private:
    socket_t sock_;

    static void init_network_stack() {
#ifdef _WIN32
        static bool initialized = false;
        if (!initialized) {
            WSADATA wsaData;
            int res = WSAStartup(MAKEWORD(2, 2), &wsaData);
            if (res == 0) {
                initialized = true;
            }
        }
#endif
    }

    void close_socket() {
        if (sock_ != INVALID_SOCKET_FD) {
#ifdef _WIN32
            closesocket(sock_);
#else
            close(sock_);
#endif
            sock_ = INVALID_SOCKET_FD;
        }
    }
};

} // namespace telemetry
} // namespace defense
