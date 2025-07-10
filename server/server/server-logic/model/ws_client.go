package model

import "github.com/fasthttp/websocket"

type Client struct {
	Conn    *websocket.Conn
	Id      string
	Address string
}
