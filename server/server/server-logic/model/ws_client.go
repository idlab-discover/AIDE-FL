package model

import "github.com/fasthttp/websocket"

type Client struct {
	Conn    *websocket.Conn `json:"conn,omitempty"`
	Id      string          `json:"id"`
	Address string          `json:"address"`
}
